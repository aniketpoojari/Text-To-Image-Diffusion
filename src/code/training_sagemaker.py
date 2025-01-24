import os
import json
import accelerate
import mlflow
from dataloader import TextImageDataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL


def setup_distributed():
    """Initialize distributed training environment for SageMaker."""
    try:
        # Get SageMaker specific env variables
        sm_hosts = json.loads(os.environ.get('SM_HOSTS'))
        sm_current_host = os.environ.get('SM_CURRENT_HOST')
        world_size = len(sm_hosts)
        rank = sm_hosts.index(sm_current_host)
        local_rank = 0  # Since we're using one GPU per instance
        
        # Set environment variables required by PyTorch distributed
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Initialize the process group
        master_addr = sm_hosts[0]
        master_port = '29500'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")

    

def training():

    rank, world_size, local_rank = setup_distributed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Retrieve environment variables
    train_size = int(os.getenv("TRAIN_SIZE", "300"))
    val_size = int(os.getenv("VAL_SIZE", "30"))
    vae_image_size = tuple(map(int, os.getenv("VAE_IMAGE_SIZE", "128,128").split(",")))
    max_length = int(os.getenv("MAX_LENGTH", "77"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    
    T = int(os.getenv("T", "300"))
    
    unet_image_size = tuple(map(int, os.getenv("UNET_IMAGE_SIZE", "16,16").split(",")))
    in_channels = int(os.getenv("IN_CHANNELS", "4"))
    out_channels = int(os.getenv("OUT_CHANNELS", "4"))
    down_block_types = tuple(os.getenv("DOWN_BLOCK_TYPES", "CrossAttnDownBlock2D,DownBlock2D,CrossAttnDownBlock2D").split(","))
    up_block_types = tuple(os.getenv("UP_BLOCK_TYPES", "CrossAttnUpBlock2D,UpBlock2D,CrossAttnUpBlock2D").split(","))
    mid_block_type = os.getenv("MID_BLOCK_TYPE", "UNetMidBlock2DCrossAttn")
    block_out_channels = tuple(map(int, os.getenv("BLOCK_OUT_CHANNELS", "64,128,256").split(",")))
    layers_per_block = int(os.getenv("LAYERS_PER_BLOCK", "2"))
    norm_num_groups = int(os.getenv("NORM_NUM_GROUPS", "32"))
    cross_attention_dim = int(os.getenv("CROSS_ATTENTION_DIM", "512"))
    attention_head_dim = int(os.getenv("ATTENTION_HEAD_DIM", "12"))
    dropout = float(os.getenv("DROPOUT", "0.1"))
    time_embedding_type = os.getenv("TIME_EMBEDDING_TYPE", "positional")
    act_fn = os.getenv("ACT_FN", "silu")

    vae_learning_rate = float(os.getenv("VAE_LEARNING_RATE", "5e-5"))
    unet_learning_rate = float(os.getenv("UNET_LEARNING_RATE", "1e-3"))
    weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-4"))
    num_epochs = int(os.getenv("NUM_EPOCHS", "10"))

    # Initialize MLflow
    if rank == 0:
        experiment_name = os.getenv("EXPERIMENT_NAME", "Training")
        run_name = os.getenv("RUN_NAME", "1st")
        registered_model_name = os.getenv("REGISTERED_MODEL_NAME", "Diffusion")
        server_uri = os.getenv("SERVER_URI", "")
        s3_mlruns_bucket = os.getenv("S3_MLRUNS_BUCKET", "")
        
        # check whetherexperiment name exists in mlflow
        mlflow.set_tracking_uri(server_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name, s3_mlruns_bucket)
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.start_run(run_name=run_name)

        mlflow.log_params({
            "train_size": train_size,
            "val_size": val_size,
            "vae_image_size": vae_image_size,
            "max_length": max_length,
            "batch_size": batch_size,
            "T": T,
            "unet_image_size": unet_image_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "down_block_types": down_block_types,
            "up_block_types": up_block_types,
            "mid_block_type": mid_block_type,
            "block_out_channels": block_out_channels,
            "layers_per_block": layers_per_block,
            "norm_num_groups": norm_num_groups,
            "cross_attention_dim": cross_attention_dim,
            "attention_head_dim": attention_head_dim,
            "dropout": dropout,
            "time_embedding_type": time_embedding_type,
            "act_fn": act_fn,
            "vae_learning_rate": vae_learning_rate,
            "unet_learning_rate": unet_learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs
        })

    # Initialize datasets
    datadir = "/opt/ml/input/data/train"
    train_dataset = TextImageDataLoader(datadir=datadir, range=(0, train_size), image_size=vae_image_size, max_text_length=max_length)
    val_dataset = TextImageDataLoader(datadir=datadir, range=(train_size, train_size + val_size), image_size=vae_image_size, max_text_length=max_length)
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Initialize models    
    noise_scheduler = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02)

    # Initialize VAE and UNet
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    diffuser = UNet2DConditionModel(
        sample_size=unet_image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        dropout=dropout,
        norm_num_groups=norm_num_groups,
        mid_block_type=mid_block_type,
        time_embedding_type=time_embedding_type,
        act_fn=act_fn
    ).to(device)

    # Wrap models with DDP
    vae = nn.parallel.DistributedDataParallel(vae, device_ids=[local_rank], output_device=local_rank)
    diffuser = nn.parallel.DistributedDataParallel(diffuser, device_ids=[local_rank], output_device=local_rank)

    # Initialize optimizers and schedulers
    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=vae_learning_rate, weight_decay=weight_decay)
    optimizer_diffuser = torch.optim.AdamW(diffuser.parameters(), lr=unet_learning_rate, weight_decay=weight_decay)
    
    scheduler_vae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vae, T_max=num_epochs)
    scheduler_diffuser = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diffuser, T_max=num_epochs)

    # Gradient scaler for mixed precision
    scaler_vae = torch.cuda.amp.GradScaler()
    scaler_diffuser = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        train_sampler.set_epoch(epoch)
        
        vae.train()
        diffuser.train()

        train_vae_epoch_loss = torch.tensor(0.0, device=device)
        train_diffuser_epoch_loss = torch.tensor(0.0, device=device)
        train_samples = torch.tensor(0, device=device)

        for images, captions, _ in train_loader:
            images = images.to(device)
            captions = captions.to(device)
            batch_size = images.shape[0]
            
            optimizer_vae.zero_grad()

            # VAE forward pass: reconstruction loss
            with torch.autocast(device_type=device, dtype=torch.float16):
                latents = vae.module.encode(images).latent_dist.sample()
                reconstructed_images = vae.module.decode(latents).sample
                reconstruction_loss = F.mse_loss(reconstructed_images, images)
            train_vae_epoch_loss += reconstruction_loss.detach() * batch_size
            
            # VAE backward pass: update parameters
            scaler_vae.scale(reconstruction_loss).backward()
            scaler_vae.unscale_(optimizer_vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            scaler_vae.step(optimizer_vae)
            scaler_vae.update()

            # Normalize latents before passing to diffuser
            latents = latents.detach() * 0.18215
            
            # Add noise
            ts = torch.randint(0, T, (latents.shape[0],), device=device)
            epsilons = torch.randn_like(latents, device=device)
            noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

            optimizer_diffuser.zero_grad()
            
            # Predict noise and calculate loss
            with torch.autocast(device_type=device, dtype=torch.float16):
                noise_pred = diffuser(noisy_latents, ts, encoder_hidden_states=captions, return_dict=False)[0]
                diffusion_loss = F.mse_loss(noise_pred, epsilons, reduction="mean")

            train_diffuser_epoch_loss += diffusion_loss.detach() * batch_size
            train_samples += batch_size
            
            # Backward pass
            scaler_diffuser.scale(diffusion_loss).backward()
            scaler_diffuser.unscale_(optimizer_diffuser)
            torch.nn.utils.clip_grad_norm_(diffuser.parameters(), max_norm=1.0)
            scaler_diffuser.step(optimizer_diffuser)
            scaler_diffuser.update()

        # Validation loop
        vae.eval()
        diffuser.eval()

        val_vae_epoch_loss = torch.tensor(0.0, device=device)
        val_diffuser_epoch_loss = torch.tensor(0.0, device=device)
        val_samples = torch.tensor(0, device=device)

        with torch.no_grad():
            for images, captions, _ in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                batch_size = images.shape[0]

                # VAE forward pass: reconstruction loss
                with torch.autocast(device_type=device, dtype=torch.float16):
                    latents = vae.module.encode(images).latent_dist.sample()
                    reconstructed_images = vae.module.decode(latents).sample
                    reconstruction_loss = F.mse_loss(reconstructed_images, images)
                val_vae_epoch_loss += reconstruction_loss.detach() * batch_size

                # Normalize latents before passing to diffuser
                latents = latents.detach() * 0.18215
                
                # Add noise
                ts = torch.randint(0, T, (latents.shape[0],), device=device)
                epsilons = torch.randn_like(latents, device=device)
                noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

                # Predict noise and calculate loss
                with torch.autocast(device_type=device, dtype=torch.float16):
                    noise_pred = diffuser(noisy_latents, ts, encoder_hidden_states=captions, return_dict=False)[0]
                    diffusion_loss = F.mse_loss(noise_pred, epsilons, reduction="mean")
                val_diffuser_epoch_loss += diffusion_loss.detach() * batch_size
                val_samples += batch_size

        # Aggregate metrics across all processes
        dist.all_reduce(train_vae_epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_diffuser_epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_vae_epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_diffuser_epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_samples, op=dist.ReduceOp.SUM)

        # Calculate final metrics
        train_vae_epoch_loss = (train_vae_epoch_loss / train_samples).item()
        train_diffuser_epoch_loss = (train_diffuser_epoch_loss / train_samples).item()
        val_vae_epoch_loss = (val_vae_epoch_loss / val_samples).item()
        val_diffuser_epoch_loss = (val_diffuser_epoch_loss / val_samples).item()

        # Log metrics to MLflow
        if rank == 0:
            mlflow.log_metric("train_vae_loss", train_vae_epoch_loss, step=epoch)
            mlflow.log_metric("train_diffuser_loss", train_diffuser_epoch_loss, step=epoch)
            mlflow.log_metric("val_vae_loss", val_vae_epoch_loss, step=epoch)
            mlflow.log_metric("val_diffuser_loss", val_diffuser_epoch_loss, step=epoch)
                
        # Update scheduler
        scheduler_vae.step()
        scheduler_diffuser.step()

        # Log epoch completion
        if rank == 0:
            print(f"Epoch {epoch + 1} - Train VAE: {train_vae_epoch_loss:.4f} | Val VAE: {val_vae_epoch_loss:.4f} | "
                  f"Train Diff: {train_diffuser_epoch_loss:.4f} | Val Diff: {val_diffuser_epoch_loss:.4f}")

    if rank == 0:
        mlflow.pytorch.log_model(vae.module, "vae", registered_model_name=registered_model_name)
        mlflow.pytorch.log_model(diffuser.module, "diffuser", registered_model_name=registered_model_name)
        mlflow.end_run()
        

if __name__ == "__main__":
    training()

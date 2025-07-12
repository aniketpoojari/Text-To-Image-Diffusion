import os
import json
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader import TextImageDataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
import deepspeed
import zipfile
import time
import boto3

def unzip_data(rank):
    """Unzip the data file in the SageMaker container."""
    try:
        input_dir = "/opt/ml/input/data/train"
        data_zip = os.path.join(input_dir, "flowers.zip")
        extract_dir = "/opt/ml/input/data/train"
        
        # Only have rank 0 print progress messages
        verbose = rank == 0
        
        if verbose:
            print(f"Checking for zipped data at {data_zip}")
        
        if os.path.exists(data_zip):
            if verbose:
                print(f"Found zipped data. Extracting to {extract_dir}....")
            
            # Create extraction directory if it doesn't exist.
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the zip file
            with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            if verbose:
                print("Extraction complete!")
                
                # List directories to confirm extraction
                print(f"Extracted contents: {os.listdir(extract_dir)}")
                if os.path.exists(os.path.join(extract_dir, "images")):
                    print(f"Number of images: {len(os.listdir(os.path.join(extract_dir, 'images')))}")
                if os.path.exists(os.path.join(extract_dir, "captions")):
                    print(f"Number of captions: {len(os.listdir(os.path.join(extract_dir, 'captions')))}")
            
            # return extract_dir
        else:
            if verbose:
                print("No zipped data found. Using original input directory.")
            # return input_dir
    
    except Exception as e:
        print(f"Error during data extraction: {e}")
        # return "/opt/ml/input/data/train"  # Fallback


def setup_distributed():
    """
    Initialize distributed training using DeepSpeed's communication module.
    On SageMaker, environment variables (SM_HOSTS, SM_CURRENT_HOST) determine the cluster configuration.
    """
    try:
        # Parse SageMaker environment variables
        sm_hosts = json.loads(os.environ.get('SM_HOSTS'))
        sm_current_host = os.environ.get('SM_CURRENT_HOST')
        world_size = len(sm_hosts)
        rank = sm_hosts.index(sm_current_host)
        local_rank = 0  # Typically one GPU per instance in this setup

        # Set the usual env variables (DeepSpeed still honors these under the hood)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

        master_addr = sm_hosts[0]
        master_port = '29500'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # Initialize distributed via DeepSpeed
        deepspeed.init_distributed()


        # Set device
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")


def training():

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Clears unused memory
    torch.cuda.empty_cache()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Environment variables
    train_size = int(os.getenv("TRAIN_SIZE", "300"))
    val_size = int(os.getenv("VAL_SIZE", "30"))
    vae_image_size = tuple(map(int, os.getenv("VAE_IMAGE_SIZE", "128,128").split(",")))
    max_length = int(os.getenv("MAX_LENGTH", "77"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    T = int(os.getenv("T", "300"))

    unet_image_size = tuple(map(int, os.getenv("UNET_IMAGE_SIZE", "16,16").split(",")))
    in_channels = int(os.getenv("IN_CHANNELS", "4"))
    out_channels = int(os.getenv("OUT_CHANNELS", "4"))
    down_block_types = tuple(os.getenv("DOWN_BLOCK_TYPES").split(","))
    up_block_types = tuple(os.getenv("UP_BLOCK_TYPES").split(","))
    mid_block_type = os.getenv("MID_BLOCK_TYPE")
    block_out_channels = tuple(map(int, os.getenv("BLOCK_OUT_CHANNELS").split(",")))
    layers_per_block = int(os.getenv("LAYERS_PER_BLOCK"))
    norm_num_groups = int(os.getenv("NORM_NUM_GROUPS"))
    cross_attention_dim = int(os.getenv("CROSS_ATTENTION_DIM"))
    attention_head_dim = int(os.getenv("ATTENTION_HEAD_DIM"))
    dropout = float(os.getenv("DROPOUT"))
    time_embedding_type = os.getenv("TIME_EMBEDDING_TYPE")
    act_fn = os.getenv("ACT_FN")

    vae_learning_rate = float(os.getenv("VAE_LEARNING_RATE"))
    unet_learning_rate = float(os.getenv("UNET_LEARNING_RATE"))
    weight_decay = float(os.getenv("WEIGHT_DECAY"))
    num_epochs = int(os.getenv("NUM_EPOCHS"))

    # MLflow setup (only by rank 0)
    if rank == 0:
        experiment_name = os.getenv("EXPERIMENT_NAME")
        run_name = os.getenv("RUN_NAME")
        registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
        server_uri = os.getenv("SERVER_URI")
        s3_mlruns_bucket = os.getenv("S3_MLRUNS_BUCKET")

        mlflow.set_tracking_uri(server_uri)
        # if mlflow.get_experiment_by_name(experiment_name) is None:
            # mlflow.create_experiment(experiment_name, s3_mlruns_bucket)
        mlflow.set_experiment(experiment_name)
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
            "num_epochs": num_epochs,
            "world_size": world_size
        })

    unzip_data(rank)

    # Load data
    datadir = "/opt/ml/input/data/train/flowers"
    train_dataset = TextImageDataLoader(datadir, range=(0, train_size), image_size=vae_image_size, max_text_length=max_length)
    val_dataset = TextImageDataLoader(datadir, range=(train_size, train_size + val_size), image_size=vae_image_size, max_text_length=max_length)

    # Use DistributedSampler for proper data sharding
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Configure DataLoader with no workers to avoid multiprocessing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    noise_scheduler = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02)

    # Create models
    model_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    model_diffuser = UNet2DConditionModel(
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
    )
    
    ## optimizer_vae = deepspeed.ops.adam.DeepSpeedCPUAdam(model_vae.parameters(), lr=vae_learning_rate, weight_decay=weight_decay)
    ## optimizer_diffuser = deepspeed.ops.adam.DeepSpeedCPUAdam(model_diffuser.parameters(), lr=unet_learning_rate, weight_decay=weight_decay)

    ds_config_vae = {
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": vae_learning_rate,
                "weight_decay": weight_decay
            }
        },
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "initial_scale_power": 16,  # Start with higher scale
            "loss_scale_window": 1000,  # Larger window
            "hysteresis": 4,            # More conservative adjustment
            "min_loss_scale": 1         # Higher minimum
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            }
        },
        "gradient_clipping": 0.5
    }

    ds_config_diffuser = ds_config_vae.copy()
    ds_config_diffuser["optimizer"]["params"]["lr"] = unet_learning_rate

    ## model_vae, optimizer_vae, _, _ = deepspeed.initialize(model=model_vae, optimizer=optimizer_vae, config=ds_config)
    ## model_diffuser, optimizer_diffuser, _, _ = deepspeed.initialize(model=model_diffuser, optimizer=optimizer_diffuser, config=ds_config)

    model_vae, optimizer_vae, _, _ = deepspeed.initialize(model=model_vae, config=ds_config_vae)
    model_diffuser, optimizer_diffuser, _, _ = deepspeed.initialize(model=model_diffuser, config=ds_config_diffuser)

    # Wrap models with DDP
    # model_vae = DDP(model_vae, device_ids=[rank])
    # model_diffuser = DDP(model_diffuser, device_ids=[rank])

    if rank == 0:
        print("\n\n\nStarting training...\n\n\n")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB", f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

    # Initialize loss tracking dictionaries
    epoch_losses = {
        'train': {'vae': [], 'diffuser': []},
        'val': {'vae': [], 'diffuser': []}
    }


    # Training loop
    for epoch in range(num_epochs):

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")

        # Set epoch for proper shuffling
        train_sampler.set_epoch(epoch)

        # Record the start time
        start_time = time.time()
        
        # ---- Training ----
        model_vae.train()
        model_diffuser.train()
        total_vae_loss = 0.0
        total_diff_loss = 0.0
        
        for images, captions, _ in train_loader:
            # Move data to device here, not in the dataloader
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                latents = model_vae.module.encode(images).latent_dist.sample()
                recon = model_vae.module.decode(latents).sample
                recon_loss = F.mse_loss(recon, images)

            model_vae.backward(recon_loss)
            model_vae.step()

            latents = latents.detach() * 0.18215  # Scale factor for VAE conditioning
            ts = torch.randint(0, T, (latents.shape[0],), device=device)
            epsilons = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

            with torch.amp.autocast('cuda'):
                noise_pred = model_diffuser(noisy_latents, ts, encoder_hidden_states=captions, return_dict=False)[0]
                diff_loss = F.mse_loss(noise_pred, epsilons)

            model_diffuser.backward(diff_loss)
            model_diffuser.step()

            # Reduce losses across processes
            ## reduced_vae_loss = reduce_tensor(recon_loss.detach(), world_size)
            ## reduced_diff_loss = reduce_tensor(diff_loss.detach(), world_size)
            
            ## total_vae_loss += reduced_vae_loss.item()
            ## total_diff_loss += reduced_diff_loss.item()
            total_vae_loss += recon_loss.item()
            total_diff_loss += diff_loss.item()

            # Clears unused memory
            # torch.cuda.empty_cache()
        
        # Aggregate losses across all steps
        avg_vae = total_vae_loss / len(train_loader)
        avg_diff = total_diff_loss / len(train_loader)

        # Collect all losses from all devices
        all_train_losses = torch.tensor([avg_vae, avg_diff], device=device)
        torch.distributed.all_reduce(all_train_losses, op=torch.distributed.ReduceOp.SUM)
        all_train_losses /= world_size

        # Store aggregated training losses
        epoch_losses['train']['vae'].append(all_train_losses[0].item())
        epoch_losses['train']['diffuser'].append(all_train_losses[1].item())

        # ---- Validation ----
        model_vae.eval()
        model_diffuser.eval()
        val_vae_loss = 0.0
        val_diff_loss = 0.0

        with torch.no_grad():
            for images, captions, _ in val_loader:
                # Move data to device here, not in the dataloader
                images = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    latents = model_vae.module.encode(images).latent_dist.sample()
                    recon = model_vae.module.decode(latents).sample
                    recon_loss = F.mse_loss(recon, images)
                
                # Reduce validation loss
                ## reduced_vae_loss = reduce_tensor(recon_loss.detach(), world_size)
                ## val_vae_loss += reduced_vae_loss.item()

                latents = latents.detach() * 0.18215
                ts = torch.randint(0, T, (latents.shape[0],), device=device)
                epsilons = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

                with torch.amp.autocast('cuda'):
                    noise_pred = model_diffuser(noisy_latents, ts, encoder_hidden_states=captions, return_dict=False)[0]
                    diff_loss = F.mse_loss(noise_pred, epsilons)
                
                # Reduce validation loss
                ## reduced_diff_loss = reduce_tensor(diff_loss.detach(), world_size)
                ## val_diff_loss += reduced_diff_loss.item()

                val_vae_loss += recon_loss.item()
                val_diff_loss += diff_loss.item()

                # Clears unused memory
                # torch.cuda.empty_cache()
            
            # Aggregate validation losses
            avg_val_vae = val_vae_loss / len(val_loader)
            avg_val_diff = val_diff_loss / len(val_loader)

            # Collect all validation losses from all devices
            all_val_losses = torch.tensor([avg_val_vae, avg_val_diff], device=device)
            torch.distributed.all_reduce(all_val_losses, op=torch.distributed.ReduceOp.SUM)
            all_val_losses /= world_size

            # Store aggregated validation losses
            epoch_losses['val']['vae'].append(all_val_losses[0].item())
            epoch_losses['val']['diffuser'].append(all_val_losses[1].item())

            # Record the end time
            end_time = time.time()

        if rank == 0:
            # Log metrics to MLflow
            mlflow.log_metric("train_vae_loss", all_train_losses[0].item(), step=epoch)
            mlflow.log_metric("train_diffuser_loss", all_train_losses[1].item(), step=epoch)
            mlflow.log_metric("val_vae_loss", all_val_losses[0].item(), step=epoch)
            mlflow.log_metric("val_diffuser_loss", all_val_losses[1].item(), step=epoch)
            
            # Print all losses in one line
            print(f"[Epoch {epoch+1}/{num_epochs}] Train: VAE={all_train_losses[0].item():.4f}, Diff={all_train_losses[1].item():.4f} | Val: VAE={all_val_losses[0].item():.4f}, Diff={all_val_losses[1].item():.4f} | Time: {(end_time - start_time):.2f}s")

    # Final model saving
    if rank == 0:
        print("\n\n\nTraining complete. Logging models...\n\n\n")
        
        # Create fresh instances of the models for saving to MLflow
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.load_state_dict(model_vae.module.state_dict()) 
        
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
        )
        diffuser.load_state_dict(model_diffuser.module.state_dict())
        
        print("Saving models to S3...")

        # Save locally
        torch.save(vae, 'vae.pth')
        torch.save(diffuser, 'diffuser.pth')

        # Upload to S3
        run_id = mlflow.active_run().info.run_id
        s3 = boto3.client('s3')

        s3.upload_file('vae.pth', s3_mlruns_bucket, f'{run_id}/vae.pth')
        s3.upload_file('diffuser.pth', s3_mlruns_bucket, f'{run_id}/diffuser.pth')

        print("Models successfully saved to S3")

        mlflow.end_run()
            
    # Clean up
    # dist.deinit_distributed()
    # deepspeed.comm.destroy_process_group()
    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    training()
import argparse
from common import read_params
from dataloader import TextImageDataLoader
import mlflow
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from datetime import datetime


def training(config_path):
    # LOAD CONFIGURATION PARAMETERS
    config = read_params(config_path)
    
    datadir = config['data']['raw']
    train_size = config['data']['train_size']
    val_size = config['data']['val_size']
    vae_image_size = tuple(config['vae']['image_size'].split(","))
    max_length = config['clip']['max_length']
    batch_size = config['training']['batch_size']

    T = config['DDPMScheduler']['T']

    unet_image_size = tuple(config['unet']['image_size'].split(","))
    in_channels = config['unet']['in_channels']
    out_channels = config['unet']['out_channels']
    down_block_types = tuple(config['unet']['down_block_types'].split(","))
    up_block_types = tuple(config['unet']['up_block_types'].split(","))
    mid_block_type = config['unet']['mid_block_type']
    block_out_channels = tuple(config['unet']['block_out_channels'].split(","))
    layers_per_block = config['unet']['layers_per_block']
    norm_num_groups = config['unet']['norm_num_groups']
    cross_attention_dim = config['unet']['cross_attention_dim']
    attention_head_dim = config['unet']['attention_head_dim'] 
    dropout = config['unet']['dropout']
    time_embedding_type = config['unet']['time_embedding_type']
    act_fn = config['unet']['act_fn']

    vae_learning_rate =  float(config['training']['vae_learning_rate'])
    unet_learning_rate = float(config['training']['unet_learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    num_epochs = config['training']['num_epochs']

    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]
    registered_model_name = config["mlflow"]["registered_model_name"]
    server_uri = config["mlflow"]["server_uri"]
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    # INITIALIZE DATASETS AND DATALOADERS
    print(datadir, train_size, val_size, vae_image_size, max_length, batch_size)
    train_dataset = TextImageDataLoader(datadir=datadir, range=(0, train_size), image_size=vae_image_size, max_text_length=max_length)
    val_dataset = TextImageDataLoader(datadir=datadir, range=(train_size, train_size + val_size), image_size=vae_image_size, max_text_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # SET FLOAT32 PRECISION FOR MATRIX OPERATIONS
    torch.set_float32_matmul_precision("medium")

    # DETERMINE DEVICE (CUDA or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # INITIALIZE NOISE SCHEDULER FOR DIFFUSION
    noise_scheduler = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4, beta_end=0.02)
    
    # INITIALIZE VAE (PRE-TRAINED MODEL)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    
    # INITIALIZE DIFFUSER (UNET-BASED MODEL)
    diffuser = UNet2DConditionModel(
        sample_size=unet_image_size,  
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        mid_block_type=mid_block_type,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        dropout=dropout,
        time_embedding_type=time_embedding_type,
        act_fn=act_fn
    ).to(device)

    # INITIALIZE OPTIMIZERS FOR VAE AND DIFFUSER
    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=vae_learning_rate, weight_decay=weight_decay)
    optimizer_diffuser = torch.optim.AdamW(diffuser.parameters(), lr=unet_learning_rate, weight_decay=weight_decay)

    # INITIALIZE LEARNING RATE SCHEDULERS
    scheduler_vae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vae, num_epochs)
    scheduler_diffuser = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diffuser, num_epochs)

    # INITIALIZE GRAD SCALERS FOR AMP (AUTOMATIC MIXED PRECISION)
    scaler_vae = torch.amp.GradScaler()
    scaler_diffuser = torch.amp.GradScaler()

    # INITIALIZE MLFLOW RUN
    with mlflow.start_run(run_name=run_name) as mlflow_run:

        # LOG HYPERPARAMETERS
        mlflow.log_params({
            "vae_learning_rate": vae_learning_rate,
            "unet_learning_rate": unet_learning_rate,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "T": T,
            "vae_image_size": vae_image_size,
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
            "act_fn": act_fn
        })

        # EPOCH LOOP: ITERATE OVER ALL EPOCHS
        for epoch in range(num_epochs):

            # TRAINING LOOP: TRAIN VAE AND DIFFUSER
            vae.train()
            diffuser.train()
            train_vae_epoch_loss = 0.0
            train_diffuser_epoch_loss = 0.0
            train_l = 0
            for images, captions, text in train_loader:
                images = images.to(device)
                captions = captions.to(device).float()

                # VAE FORWARD PASS: RECONSTRUCTION LOSS
                with torch.autocast(device_type=device, dtype=torch.bfloat16):  # ENABLE MIXED PRECISION
                    latents = vae.encode(images).latent_dist.sample()
                    reconstructed_images = vae.decode(latents).sample
                    reconstruction_loss = F.mse_loss(reconstructed_images, images)
                train_vae_epoch_loss += reconstruction_loss.item() * images.shape[0]
                train_l += images.shape[0]

                # VAE BACKWARD PASS: UPDATE PARAMETERS
                optimizer_vae.zero_grad()
                torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                scaler_vae.scale(reconstruction_loss).backward()  # SCALING LOSS FOR AMP
                scaler_vae.step(optimizer_vae)  # APPLY GRADIENTS
                scaler_vae.update()  # UPDATE GRAD SCALER

                # NORMALIZE LATENTS BEFORE PASSING TO DIFFUSER
                latents = latents.detach() * 0.18215

                # DIFFUSER FORWARD PASS: ADD NOISE AND DENOISE
                ts = torch.randint(0, T, [latents.shape[0]], device=device)
                epsilons = torch.randn_like(latents, device=device)
                noise_imgs = noise_scheduler.add_noise(latents, epsilons, ts)

                with torch.autocast(device_type=device, dtype=torch.bfloat16):  # ENABLE MIXED PRECISION
                    noise_pred = diffuser(noise_imgs, ts, encoder_hidden_states=captions, return_dict=False)[0]
                    diffusion_loss = F.mse_loss(noise_pred, epsilons)
                train_diffuser_epoch_loss += diffusion_loss.item() * images.shape[0]

                # DIFFUSER BACKWARD PASS: UPDATE PARAMETERS
                optimizer_diffuser.zero_grad()
                torch.nn.utils.clip_grad_norm_(diffuser.parameters(), max_norm=1.0)
                scaler_diffuser.scale(diffusion_loss).backward()  # SCALING LOSS FOR AMP
                scaler_diffuser.step(optimizer_diffuser)  # APPLY GRADIENTS
                scaler_diffuser.update()  # UPDATE GRAD SCALER

            # VALIDATION LOOP: EVALUATE PERFORMANCE ON VALIDATION SET
            vae.eval()
            diffuser.eval()
            val_vae_epoch_loss = 0.0
            val_diffuser_epoch_loss = 0.0
            val_l = 0
            with torch.no_grad():
                for images, captions, text in val_loader:
                    images = images.to(device)
                    captions = captions.to(device).float()

                    # VAE EVALUATION
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):  # ENABLE MIXED PRECISION
                        latents = vae.encode(images).latent_dist.sample()
                        reconstructed_images = vae.decode(latents).sample
                        reconstruction_loss = F.mse_loss(reconstructed_images, images)
                    val_vae_epoch_loss += reconstruction_loss.item() * images.shape[0]
                    val_l += images.shape[0]

                    # NORMALIZE LATENTS BEFORE PASSING TO DIFFUSER
                    latents = latents.detach() * 0.18215

                    # DIFFUSER EVALUATION
                    ts = torch.randint(0, T, [latents.shape[0]], device=device)
                    epsilons = torch.randn_like(latents, device=device)
                    noise_imgs = noise_scheduler.add_noise(latents, epsilons, ts)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16):  # ENABLE MIXED PRECISION
                        noise_pred = diffuser(noise_imgs, ts, encoder_hidden_states=captions, return_dict=False)[0]
                        diffusion_loss = F.mse_loss(noise_pred, epsilons)
                    val_diffuser_epoch_loss += diffusion_loss.item() * images.shape[0]

            # STEP LEARNING RATE SCHEDULERS
            scheduler_vae.step()
            scheduler_diffuser.step()
            
            # CALCULATE AVERAGE LOSSES FOR TRAINING AND VALIDATION
            avg_train_vae_epoch_loss = train_vae_epoch_loss / train_l
            avg_val_vae_epoch_loss = val_vae_epoch_loss / val_l
            avg_train_diffuser_epoch_loss = train_diffuser_epoch_loss / train_l
            avg_val_diffuser_epoch_loss = val_diffuser_epoch_loss / val_l

            # LOG TRAINING AND VALIDATION LOSS TO MLFLOW
            mlflow.log_metric("train_vae_loss", avg_train_vae_epoch_loss, epoch + 1)
            mlflow.log_metric("val_vae_loss", avg_val_vae_epoch_loss, epoch + 1)
            mlflow.log_metric("train_diffuser_loss", avg_train_diffuser_epoch_loss, epoch + 1)
            mlflow.log_metric("val_diffuser_loss", avg_val_diffuser_epoch_loss, epoch + 1)


            # STEP LEARNING RATE SCHEDULERS BASED ON EVALUATION LOSS
            scheduler_vae.step(avg_val_vae_epoch_loss)
            scheduler_diffuser.step(avg_val_diffuser_epoch_loss)

            # PRINT TRAINING AND VALIDATION STATS
            print(f"Epoch {epoch + 1} - Train VAE: {avg_train_vae_epoch_loss:.4f} | Val VAE: {avg_val_vae_epoch_loss:.4f} | Train Diff: {avg_train_diffuser_epoch_loss:.4f} | Val Diff: {avg_val_diffuser_epoch_loss:.4f}")

        # SAVE MODELS TO MLFLOW
        mlflow.pytorch.log_model(vae, f"{mlflow_run.info.run_id}/vae", registered_model_name=registered_model_name)
        mlflow.pytorch.log_model(diffuser, f"{mlflow_run.info.run_id}/diffuser", registered_model_name=registered_model_name)

    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("training_completion.txt", "w") as file:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    # PARSE CONFIG FILE ARGUMENT
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    # START TRAINING PROCESS
    training(config_path=parsed_args.config)

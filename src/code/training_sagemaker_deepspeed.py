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


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters. Using EMA weights at inference
    produces significantly smoother, higher-quality generated images than the
    raw training weights.

    decay=0.9999 is standard for diffusion models (effectively averages over the
    last ~10,000 gradient steps).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Store shadow params on CPU to avoid consuming GPU memory
        self.shadow = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.cpu(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module):
        """Copy EMA weights into the model (for saving or evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: torch.nn.Module, backup: dict):
        """Restore original (non-EMA) weights from a backup dict."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name].to(param.device))


# ── Helpers ───────────────────────────────────────────────────────────────────

def unzip_data(rank):
    """Unzip the data file in the SageMaker container."""
    try:
        input_dir = "/opt/ml/input/data/train"
        data_zip = os.path.join(input_dir, "flowers.zip")
        extract_dir = "/opt/ml/input/data/train"
        verbose = rank == 0

        if verbose:
            print(f"Checking for zipped data at {data_zip}")

        if os.path.exists(data_zip):
            if verbose:
                print(f"Found zipped data. Extracting to {extract_dir}...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(data_zip, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            if verbose:
                print("Extraction complete!")
                print(f"Extracted contents: {os.listdir(extract_dir)}")
                images_dir = os.path.join(extract_dir, "images")
                captions_dir = os.path.join(extract_dir, "captions")
                if os.path.exists(images_dir):
                    print(f"Number of images: {len(os.listdir(images_dir))}")
                if os.path.exists(captions_dir):
                    print(f"Number of captions: {len(os.listdir(captions_dir))}")
        else:
            if verbose:
                print("No zipped data found. Using original input directory.")

    except Exception as e:
        print(f"Error during data extraction: {e}")


def setup_distributed():
    """Initialize distributed training using DeepSpeed's communication module."""
    try:
        sm_hosts = json.loads(os.environ.get("SM_HOSTS"))
        sm_current_host = os.environ.get("SM_CURRENT_HOST")

        rank = sm_hosts.index(sm_current_host)
        local_rank = 0  # One GPU per instance

        os.environ.update({
            "WORLD_SIZE": str(len(sm_hosts)),
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "MASTER_ADDR": sm_hosts[0],
            "MASTER_PORT": "29500",
        })

        deepspeed.init_distributed()
        torch.cuda.set_device(local_rank)

        return rank, len(sm_hosts), local_rank

    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed training: {e}")


# ── Training ──────────────────────────────────────────────────────────────────

def training():

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # auto-tune conv algorithms for fixed input shapes

    # Environment variables
    train_size = int(os.getenv("TRAIN_SIZE", "300"))
    val_size   = int(os.getenv("VAL_SIZE", "30"))
    vae_image_size = tuple(map(int, os.getenv("VAE_IMAGE_SIZE", "128,128").split(",")))
    max_length = int(os.getenv("MAX_LENGTH", "77"))
    batch_size = int(os.getenv("BATCH_SIZE", "4"))
    T          = int(os.getenv("T", "1000"))

    unet_image_size   = tuple(map(int, os.getenv("UNET_IMAGE_SIZE", "16,16").split(",")))
    in_channels       = int(os.getenv("IN_CHANNELS", "4"))
    out_channels      = int(os.getenv("OUT_CHANNELS", "4"))
    down_block_types  = tuple(os.getenv("DOWN_BLOCK_TYPES").split(","))
    up_block_types    = tuple(os.getenv("UP_BLOCK_TYPES").split(","))
    mid_block_type    = os.getenv("MID_BLOCK_TYPE")
    block_out_channels = tuple(map(int, os.getenv("BLOCK_OUT_CHANNELS").split(",")))
    layers_per_block  = int(os.getenv("LAYERS_PER_BLOCK"))
    norm_num_groups   = int(os.getenv("NORM_NUM_GROUPS"))
    cross_attention_dim = int(os.getenv("CROSS_ATTENTION_DIM"))
    attention_head_dim  = int(os.getenv("ATTENTION_HEAD_DIM"))
    dropout             = float(os.getenv("DROPOUT"))
    time_embedding_type = os.getenv("TIME_EMBEDDING_TYPE")
    act_fn              = os.getenv("ACT_FN")

    unet_learning_rate = float(os.getenv("UNET_LEARNING_RATE"))
    weight_decay       = float(os.getenv("WEIGHT_DECAY"))
    num_epochs         = int(os.getenv("NUM_EPOCHS"))

    # MLflow setup (rank 0 only)
    if rank == 0:
        experiment_name      = os.getenv("EXPERIMENT_NAME")
        run_name             = os.getenv("RUN_NAME")
        registered_model_name = os.getenv("REGISTERED_MODEL_NAME")
        server_uri           = os.getenv("SERVER_URI")
        s3_mlruns_bucket     = os.getenv("S3_MLRUNS_BUCKET")

        mlflow.set_tracking_uri(server_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "train_size": train_size, "val_size": val_size,
            "vae_image_size": vae_image_size, "max_length": max_length,
            "batch_size": batch_size, "T": T,
            "unet_image_size": unet_image_size,
            "in_channels": in_channels, "out_channels": out_channels,
            "down_block_types": down_block_types, "up_block_types": up_block_types,
            "mid_block_type": mid_block_type, "block_out_channels": block_out_channels,
            "layers_per_block": layers_per_block, "norm_num_groups": norm_num_groups,
            "cross_attention_dim": cross_attention_dim,
            "attention_head_dim": attention_head_dim,
            "dropout": dropout, "time_embedding_type": time_embedding_type,
            "act_fn": act_fn,
            "unet_learning_rate": unet_learning_rate,
            "weight_decay": weight_decay, "num_epochs": num_epochs,
            "world_size": world_size, "ema_decay": 0.9999,
        })

    unzip_data(rank)

    # ── Datasets ──────────────────────────────────────────────────────────────
    # range=(start, end) semantics — the dataloader bug fix is in dataloader.py
    datadir = "/opt/ml/input/data/train/flowers"
    train_dataset = TextImageDataLoader(
        datadir, range=(0, train_size),
        image_size=vae_image_size, max_text_length=max_length,
    )
    val_dataset = TextImageDataLoader(
        datadir, range=(train_size, train_size + val_size),
        image_size=vae_image_size, max_text_length=max_length,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=T,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
    )

    # ── Models ────────────────────────────────────────────────────────────────
    model_vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
    ).to(device)
    model_vae.eval()

    model_diffuser = UNet2DConditionModel(
        sample_size=unet_image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        mid_block_type=mid_block_type,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim,
        dropout=dropout,
        norm_num_groups=norm_num_groups,
        time_embedding_type=time_embedding_type,
        act_fn=act_fn,
    )

    ds_config_diffuser = {
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": unet_learning_rate,
                "betas": [0.9, 0.999],
                "weight_decay": weight_decay,
                "eps": 1e-8,
            },
        },
        "fp16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 500_000_000,
            "allgather_bucket_size": 500_000_000,
            "allgather_partitions": True,
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 1e-6,
                "warmup_max_lr": unet_learning_rate,
                "warmup_num_steps": 500,
            },
        },
    }

    model_diffuser, _, _, _ = deepspeed.initialize(
        model=model_diffuser, config=ds_config_diffuser
    )

    # EMA — maintained on rank 0 only (rank 0 has all parameters with ZeRO-2)
    ema = EMA(model_diffuser.module, decay=0.9999) if rank == 0 else None

    PATIENCE = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    n_batches = len(train_loader)

    if rank == 0:
        print(f"\nStarting training — {num_epochs} epochs, {world_size} GPUs, early stopping patience={PATIENCE}")
        print(f"Batches/epoch (per GPU): {n_batches}  |  Global imgs/epoch: {train_size}")
        print(
            f"GPU memory — Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB  "
            f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB\n"
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        model_vae.eval()
        model_diffuser.train()
        total_epoch_diff_loss = 0.0

        acc_data = acc_vae = acc_unet = acc_bwd = 0.0

        LOG_EVERY = 10
        global_batches_done = epoch * n_batches
        total_batches_all = n_batches * num_epochs

        data_start = time.time()

        for batch_idx, (images, captions) in enumerate(train_loader):
            torch.cuda.synchronize()
            t_data = time.time() - data_start

            batch_start = time.time()

            images   = images.to(device, dtype=torch.float16, non_blocking=True)
            captions = captions.to(device, dtype=torch.float16, non_blocking=True)

            # ── VAE encode ────────────────────────────────────────────────────
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                latents = model_vae.encode(images).latent_dist.sample()
            torch.cuda.synchronize()
            t_vae = time.time() - t0

            latents = latents.detach() * 0.18215
            ts = torch.randint(0, T, (latents.shape[0],), device=device)
            epsilons = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

            # CFG dropout: 10% unconditional training
            if torch.rand(1).item() < 0.1:
                encoder_hidden_states = (
                    train_loader.dataset.get_null_embedding(captions.shape[0])
                    .to(device, dtype=torch.float16)
                )
            else:
                encoder_hidden_states = captions

            # ── UNet forward ──────────────────────────────────────────────────
            torch.cuda.synchronize()
            t0 = time.time()
            noise_pred = model_diffuser(
                noisy_latents, ts,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            torch.cuda.synchronize()
            t_unet = time.time() - t0

            batch_diff_loss = F.mse_loss(noise_pred, epsilons)

            # ── Backward + step ───────────────────────────────────────────────
            torch.cuda.synchronize()
            t0 = time.time()
            model_diffuser.backward(batch_diff_loss)
            model_diffuser.step()
            torch.cuda.synchronize()
            t_bwd = time.time() - t0

            if rank == 0:
                ema.update(model_diffuser.module)

            total_epoch_diff_loss += batch_diff_loss.item()
            acc_data += t_data
            acc_vae  += t_vae
            acc_unet += t_unet
            acc_bwd  += t_bwd

            torch.cuda.synchronize()
            t_batch = time.time() - batch_start
            imgs_per_sec = batch_size * world_size / t_batch

            if rank == 0:
                global_batches_done += 1
                overall_pct = 100.0 * global_batches_done / total_batches_all

                if (batch_idx + 1) % LOG_EVERY == 0 or batch_idx == n_batches - 1:
                    print(
                        f"  Ep {epoch+1:>3}/{num_epochs} | "
                        f"Batch {batch_idx+1:>4}/{n_batches} | "
                        f"Overall {overall_pct:5.1f}% | "
                        f"Loss {batch_diff_loss.item():.4f} | "
                        f"data {t_data*1e3:5.1f}ms  "
                        f"vae {t_vae*1e3:5.1f}ms  "
                        f"unet {t_unet*1e3:5.1f}ms  "
                        f"bwd {t_bwd*1e3:5.1f}ms  "
                        f"batch {t_batch*1e3:6.1f}ms | "
                        f"{imgs_per_sec:5.1f} imgs/s"
                    )

            data_start = time.time()

        avg_train_loss = total_epoch_diff_loss / n_batches

        # Reduce train loss across ranks
        train_loss_t = torch.tensor([avg_train_loss], device=device)
        torch.distributed.all_reduce(train_loss_t, op=torch.distributed.ReduceOp.SUM)
        train_loss_t /= world_size

        # ── Validation ────────────────────────────────────────────────────────
        model_vae.eval()
        model_diffuser.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images   = images.to(device, dtype=torch.float16, non_blocking=True)
                captions = captions.to(device, dtype=torch.float16, non_blocking=True)

                latents = model_vae.encode(images).latent_dist.sample()
                latents = latents.detach() * 0.18215
                ts = torch.randint(0, T, (latents.shape[0],), device=device)
                epsilons = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

                noise_pred = model_diffuser(
                    noisy_latents, ts,
                    encoder_hidden_states=captions,
                    return_dict=False,
                )[0]
                total_val_loss += F.mse_loss(noise_pred, epsilons).item()

        avg_val_loss = total_val_loss / len(val_loader)

        val_loss_t = torch.tensor([avg_val_loss], device=device)
        torch.distributed.all_reduce(val_loss_t, op=torch.distributed.ReduceOp.SUM)
        val_loss_t /= world_size

        epoch_time = time.time() - epoch_start

        if rank == 0:
            mlflow.log_metric("train_diffuser_loss", train_loss_t.item(), step=epoch)
            mlflow.log_metric("val_diffuser_loss",   val_loss_t.item(),   step=epoch)
            print(
                f"\n  ── Epoch {epoch+1}/{num_epochs} summary ──\n"
                f"  Train loss : {train_loss_t.item():.4f}  |  Val loss: {val_loss_t.item():.4f}\n"
                f"  Epoch time : {epoch_time:.1f}s\n"
                f"  Avg/batch  — "
                f"data {acc_data/n_batches*1e3:.1f}ms  "
                f"vae {acc_vae/n_batches*1e3:.1f}ms  "
                f"unet {acc_unet/n_batches*1e3:.1f}ms  "
                f"bwd {acc_bwd/n_batches*1e3:.1f}ms  "
                f"total {(acc_data+acc_vae+acc_unet+acc_bwd)/n_batches*1e3:.1f}ms\n"
                f"  Bottleneck : {max(('data',acc_data),('vae',acc_vae),('unet',acc_unet),('bwd',acc_bwd),key=lambda x:x[1])[0].upper()}\n"
                f"  GPU memory — Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB  "
                f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB\n"
                f"  Early stop — best_val: {best_val_loss:.4f}  patience: {epochs_without_improvement}/{PATIENCE}\n"
            )

        # ── Early stopping (broadcast decision to all ranks) ──────────────────
        should_stop = torch.tensor([0], device=device)
        if rank == 0:
            if val_loss_t.item() < best_val_loss:
                best_val_loss = val_loss_t.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    should_stop[0] = 1
                    print(f"\n  Early stopping: val loss hasn't improved for {PATIENCE} epochs (best={best_val_loss:.4f})")

        torch.distributed.broadcast(should_stop, src=0)
        if should_stop[0] == 1:
            break

    # ── Save EMA model ────────────────────────────────────────────────────────
    if rank == 0:
        print("\nTraining complete. Saving EMA model weights...")

        # Rebuild a fresh model and load EMA weights into it
        diffuser = UNet2DConditionModel(
            sample_size=unet_image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            mid_block_type=mid_block_type,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            time_embedding_type=time_embedding_type,
            act_fn=act_fn,
        )
        diffuser.load_state_dict(model_diffuser.module.state_dict())
        # Overwrite with EMA weights — these produce better generation quality
        ema.apply_shadow(diffuser)

        torch.save(diffuser, "diffuser.pth")

        s3 = boto3.client("s3")
        run_id = mlflow.active_run().info.run_id
        s3.upload_file("diffuser.pth", s3_mlruns_bucket, f"{run_id}/diffuser.pth")
        print(f"EMA model saved to s3://{s3_mlruns_bucket}/{run_id}/diffuser.pth")

        mlflow.end_run()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if hasattr(model_diffuser, "destroy"):
        model_diffuser.destroy()
    del model_diffuser, model_vae
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    training()

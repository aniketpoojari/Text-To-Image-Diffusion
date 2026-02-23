# System Design

## Overview

This document describes the architecture of an end-to-end text-to-image generation system built around latent diffusion. The system covers the full ML lifecycle: data preparation, distributed training on cloud infrastructure, experiment tracking, model optimization, and automated deployment.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DVC Pipeline (9 stages)                         │
│                                                                        │
│  Data Prep          Training              Post-Training     Deploy     │
│  ─────────          ────────              ──────────────     ──────     │
│  Captions  ──►  S3 Upload ──►  SageMaker  ──►  Evaluate  ──► HF Hub  │
│  Embeddings          │        (DeepSpeed)       ONNX          │       │
│                      │         MLflow           TensorRT      ▼       │
│                      │                                    HF Spaces   │
│                      │                                   (CI/CD)      │
└──────────────────────┼──────────────────────────────────────────────────┘
                       │
                   params.yaml (single config source)
```

---

## 1. Data Pipeline

### Dataset

- **Source**: Oxford 102 Flowers (~8,189 images across 102 categories)
- **Caption generation**: Florence-2-large auto-generates detailed natural language descriptions for each image, since the original dataset only has category labels — not the text prompts needed for text-conditioned generation.

### CLIP Embedding Precomputation

```
Image Caption (text)
       │
       ▼
CLIP ViT-L/14 Tokenizer → Token IDs
       │
       ▼
CLIP Text Encoder (frozen) → [77 × 768] embedding
       │
       ▼
Saved as .pt file (one per image)
```

**Why precompute?** CLIP inference was the per-batch training bottleneck. Loading a cached `.pt` file is ~100× faster than running CLIP forward pass each time. This also enables `num_workers > 0` in the DataLoader since there's no GPU dependency in `__getitem__`.

A null embedding (all zeros input) is also precomputed and cached — this is used during training for the 10% unconditional dropout that enables classifier-free guidance.

### S3 Upload

The full dataset (images + captions + precomputed embeddings) is zipped and uploaded to S3 for SageMaker to access during training.

---

## 2. Model Architecture

### Latent Diffusion

Instead of operating directly on 128×128×3 pixel space, the model works in a compressed 16×16×4 latent space via a pretrained VAE. This reduces compute by ~64× compared to pixel-space diffusion.

```
                        TRAINING
                        ════════

Text ──► CLIP (frozen) ──► text_embeddings [77×768]
                                    │
Image ──► VAE Encoder ──► latent ──►│──► Add noise (DDPM, t ~ U[0,1000])
          (frozen)       [16×16×4]  │         │
                                    │         ▼
                                    └──► UNet2D (cross-attn) ──► predicted noise
                                              │
                                         MSE loss vs actual noise
                                              │
                                         Backprop (UNet only)


                       INFERENCE
                       ═════════

Text ──► CLIP ──► text_embeddings ────────────────┐
                                                   │
Null ──► CLIP ──► uncond_embeddings ──────────┐    │
                                               │    │
Random noise [16×16×4] ──► DDIM Loop (30 steps):   │
                           │                        │
                           ├── UNet(noise, t, uncond) ──► noise_uncond
                           ├── UNet(noise, t, cond)  ──► noise_cond
                           ├── noise_pred = noise_uncond + cfg_scale × (noise_cond - noise_uncond)
                           └── denoise step
                                    │
                              Clean latent
                                    │
                              VAE Decoder ──► Image [128×128×3]
```

### Component Details

| Component | Model | Trainable? | Purpose |
|---|---|---|---|
| Text Encoder | `openai/clip-vit-large-patch14` | No (frozen) | Converts text prompts to 768-dim embeddings |
| VAE | `stabilityai/sd-vae-ft-mse` | No (frozen) | 8× spatial compression to/from latent space |
| UNet | `UNet2DConditionModel` | **Yes** | Predicts noise in latent space, conditioned on text |
| Scheduler | DDPM (train) / DDIM (infer) | N/A | Controls noise addition and removal |

### UNet Architecture

```
Input: noisy_latent [B, 4, 16, 16] + timestep + text_embeddings [B, 77, 768]

Down path:
  DownBlock2D          (4 → 192 channels)
  CrossAttnDownBlock2D (192 → 384, attends to text)
  CrossAttnDownBlock2D (384 → 576, attends to text)

Mid:
  UNetMidBlock2DCrossAttn (576 channels, self-attn + cross-attn)

Up path:
  CrossAttnUpBlock2D (576 → 384, attends to text)
  CrossAttnUpBlock2D (384 → 192, attends to text)
  UpBlock2D          (192 → 4 channels)

Output: predicted_noise [B, 4, 16, 16]
```

Cross-attention layers allow the UNet to attend to text embeddings at each spatial resolution, enabling text-guided denoising.

### Classifier-Free Guidance (CFG)

During training, 10% of samples have their text embeddings replaced with a null embedding. At inference, the model runs two forward passes per step (conditional + unconditional) and blends:

```
noise_pred = noise_uncond + cfg_scale × (noise_cond - noise_uncond)
```

Higher `cfg_scale` increases prompt adherence at the cost of diversity. Default: 5.0–7.5.

---

## 3. Training Infrastructure

### Distributed Training on SageMaker

```
┌────────────────────────────────────────────────┐
│               AWS SageMaker                     │
│                                                 │
│   ┌─────────────────┐  ┌─────────────────┐     │
│   │  Node 0 (T4)    │  │  Node 1 (T4)    │     │
│   │                 │  │                 │     │
│   │  DeepSpeed      │◄─►  DeepSpeed      │     │
│   │  Rank 0         │  │  Rank 1         │     │
│   │                 │  │                 │     │
│   │  Optimizer      │  │  Optimizer      │     │
│   │  states (½)     │  │  states (½)     │     │
│   │  Gradients (½)  │  │  Gradients (½)  │     │
│   └────────┬────────┘  └────────┬────────┘     │
│            │                    │               │
│            └──────┬─────────────┘               │
│                   ▼                             │
│           S3 (best checkpoint)                  │
│                   │                             │
│                   ▼                             │
│           MLflow on DagShub                     │
│           (metrics + hyperparams)               │
└────────────────────────────────────────────────┘
```

### Why DeepSpeed ZeRO Stage 2?

| Strategy | Optimizer States | Gradients | Parameters | Memory per GPU |
|---|---|---|---|---|
| Standard DataParallel | Full copy | Full copy | Full copy | ~3× model size |
| **ZeRO Stage 2** | **Partitioned** | **Partitioned** | Full copy | ~1.5× model size |
| ZeRO Stage 3 | Partitioned | Partitioned | Partitioned | ~1× model size |

ZeRO Stage 2 was chosen as the sweet spot — it halves memory for optimizer states and gradients (the largest consumers with AdamW) while avoiding the communication overhead of parameter partitioning in Stage 3.

### Training Loop

1. Load precomputed CLIP embedding from disk (`.pt` file)
2. Load image → VAE encode → latent `[4, 16, 16]`
3. Sample random timestep `t ~ U[0, 1000]`
4. Add noise according to DDPM schedule
5. UNet predicts noise, conditioned on text embedding
6. MSE loss between predicted and actual noise
7. Backprop through UNet only (CLIP and VAE are frozen)
8. DeepSpeed handles gradient sync + optimizer step across nodes

### Experiment Tracking

MLflow on DagShub logs per-epoch:
- `train_diffuser_loss`, `val_diffuser_loss`
- Gradient norms (for monitoring training stability)
- All hyperparameters from `params.yaml`
- Best model checkpoint registered to MLflow model registry

---

## 4. Conditional Model Promotion

A key design choice: downstream stages only run when the model actually improves.

```
log_training_model stage:
    │
    ├── Fetch best run from MLflow (lowest val_diffuser_loss)
    ├── Compare against saved_models/model_metadata.json
    │
    ├── If improved:
    │   ├── Download checkpoint from S3
    │   ├── Update model_metadata.json (new run_id + loss)
    │   └── File hash changes → DVC triggers downstream stages
    │
    └── If not improved:
        └── Files unchanged → DVC skips evaluate, onnx, trt, push
```

This prevents unnecessary ONNX re-exports, TensorRT recompilation, and HF Hub uploads when training doesn't beat the previous best.

---

## 5. Model Optimization

### ONNX Export

All three model components are exported to ONNX (opset 17):

| Model | Input | Output | Purpose |
|---|---|---|---|
| `clip_text_encoder.onnx` | token_ids, attention_mask | embeddings [77×768] | Text encoding |
| `unet.onnx` | latents, timestep, text_embeddings | noise prediction | Denoising |
| `vae_decoder.onnx` | latents [16×16×4] | image [128×128×3] | Latent → pixel |

ONNX Runtime applies graph optimizations (operator fusion, constant folding) that make CPU inference significantly faster than PyTorch eager mode — critical for the free-tier HF Spaces deployment.

### TensorRT Compilation

For GPU deployment, ONNX models are further compiled to TensorRT engines:
- Fixed input shapes → kernel auto-tuning
- Layer fusion + precision calibration
- ~2–3× speedup over ONNX Runtime on NVIDIA GPUs

---

## 6. Inference Pipeline

```
User prompt ──► Tokenize (CLIP tokenizer, max_length=77)
                    │
              ┌─────┴─────┐
              ▼            ▼
         Conditional   Unconditional
         embedding     embedding (null)
              │            │
              ▼            ▼
         Random noise [16×16×4]
              │
              ▼
     ┌──── DDIM Loop (30-50 steps) ────┐
     │                                  │
     │  For each step t:                │
     │    noise_u = UNet(x, t, uncond)  │
     │    noise_c = UNet(x, t, cond)    │
     │    noise = noise_u + cfg × Δ     │
     │    x = ddim_step(x, noise, t)    │
     │                                  │
     └──────────────────────────────────┘
              │
              ▼
         VAE Decoder ──► RGB Image [128×128]
```

### Inference Backends

The Streamlit app supports three backends, selectable at runtime:

| Backend | Device | Speed (per image) | Use Case |
|---|---|---|---|
| ONNX Runtime | CPU | ~2–4 min | HF Spaces (free tier) |
| TensorRT | GPU | ~5–10 s | Local with NVIDIA GPU |
| PyTorch | CPU/GPU | Varies | Development / debugging |

---

## 7. Deployment Architecture

```
┌──────────────┐     git push main     ┌────────────────────┐
│   GitHub     │ ──────────────────►   │  GitHub Actions    │
│   (source)   │   (app.py changed)    │  CI/CD workflow    │
└──────────────┘                       └────────┬───────────┘
                                                │
                                    Upload app.py, Dockerfile,
                                       requirements.txt
                                                │
                                                ▼
                                  ┌────────────────────────┐
                                  │   HF Spaces            │
                                  │   (Docker container)   │
                                  │                        │
                                  │   1. Build from        │
                                  │      Dockerfile        │
                                  │   2. Download ONNX     │
                                  │      from HF Hub       │
                                  │   3. Serve Streamlit   │
                                  │      app on CPU        │
                                  └────────────────────────┘
                                                │
                                     Downloads at startup
                                                │
                                                ▼
                                  ┌────────────────────────┐
                                  │   HF Hub Model Repo    │
                                  │                        │
                                  │   ├── diffuser.pth     │
                                  │   └── onnx/            │
                                  │       ├── clip.onnx    │
                                  │       ├── unet.onnx    │
                                  │       └── vae.onnx     │
                                  └────────────────────────┘
```

### Why separate model repo from app deployment?

- **Model repo** (HF Hub): Updated only when the model improves (via DVC pipeline)
- **App repo** (HF Spaces): Updated on every relevant code change (via GitHub Actions)

This decouples model updates from UI updates. The app always downloads the latest model from HF Hub at startup.

---

## 8. Configuration Management

All hyperparameters and infrastructure settings live in a single `params.yaml` file, which serves as the source of truth for both DVC and the training script.

```
params.yaml
    │
    ├── data.*          → caption generator, data upload
    ├── clip.*          → embedding precomputation, dataloader
    ├── vae.*           → image preprocessing
    ├── DDPMScheduler.* → noise schedule
    ├── unet.*          → model architecture
    ├── training.*      → optimizer, epochs, batch size
    ├── mlflow.*        → experiment tracking
    ├── pytorch_estimator.* → SageMaker instance config
    └── huggingface.*   → HF Hub repo
```

`params.yaml` is git-ignored (contains credentials). A `params.yaml.template` is committed with placeholder values.

DVC tracks parameter dependencies per-stage — changing a training hyperparameter only re-runs training + downstream, not data prep.

---

## 9. Design Trade-offs

| Decision | Alternative | Why this choice |
|---|---|---|
| Latent diffusion (VAE) | Pixel-space diffusion | 64× fewer pixels to denoise, fits on T4 GPUs |
| Frozen CLIP + VAE | Fine-tune end-to-end | Drastically reduces trainable parameters; pretrained representations are already strong |
| DDIM at inference | DDPM (1000 steps) | 20–33× speedup, critical for CPU deployment |
| Precomputed embeddings | On-the-fly CLIP inference | Eliminates training bottleneck, enables multi-worker loading |
| DeepSpeed ZeRO-2 | ZeRO-3, FSDP | Best memory/communication trade-off for 2-node setup |
| ONNX for deployment | PyTorch, TorchScript | Fastest CPU inference due to graph optimization |
| Conditional promotion | Always redeploy | Prevents unnecessary ONNX exports and HF uploads |
| DVC pipeline | Airflow, Prefect | Lightweight, git-native, perfect for single-pipeline ML projects |
| Florence-2 captions | Manual labeling, BLIP | High-quality detailed captions with zero human effort |
| Spot instances | On-demand | ~60–70% cost reduction for fault-tolerant training |

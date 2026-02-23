# Text-to-Image Generation with Diffusion Models

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?logo=pytorch)](https://pytorch.org)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO_Stage_2-blue)](https://deepspeed.ai)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?logo=dvc)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-DagShub-0194E2?logo=mlflow)](https://dagshub.com)
[![ONNX](https://img.shields.io/badge/ONNX-Optimized_Inference-grey?logo=onnx)](https://onnx.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![HF Spaces](https://img.shields.io/badge/HF_Spaces-Live_Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/aniketp2009gmail/text-to-image-flowers)

An end-to-end text-to-image diffusion model — from data preparation and caption generation through distributed training on AWS SageMaker, experiment tracking with MLflow, model optimization (ONNX / TensorRT), to automated deployment on Hugging Face Spaces via CI/CD.

**[Live Demo](https://huggingface.co/spaces/aniketp2009gmail/text-to-image-flowers)** · **[Model on HF Hub](https://huggingface.co/aniketp2009gmail/flower-diffusion)** · **[Experiment Tracking](https://dagshub.com/aniketpoojari/Text-To-Image-Diffusion.mlflow)** · **[System Design](SYSTEM_DESIGN.md)**

---

## Sample Outputs

Generated with DDIM · 50 steps · CFG 5.0 · avg 2.37 s/image (GPU):

| | | | | |
|:---:|:---:|:---:|:---:|:---:|
| ![](samples/sample_1.png) | ![](samples/sample_2.png) | ![](samples/sample_3.png) | ![](samples/sample_4.png) | ![](samples/sample_5.png) |
| *Yellow sunflower* | *Red rose* | *Purple lavender* | *White daisy* | *Cherry blossom* |

---

## Pipeline Overview

The entire workflow — data prep, training, evaluation, export, and deployment — is orchestrated as a single reproducible DVC pipeline.

```mermaid
graph TD
    A[Oxford 102 Flowers\n~8K images] --> B[Florence-2-large\nCaption Generator]
    B --> B2[CLIP Embedding\nPrecomputation]
    B2 --> C[Upload to S3]
    C --> D[AWS SageMaker\nDistributed Training]
    D --> E[DeepSpeed ZeRO Stage 2\nFP16 Mixed Precision]
    E --> F[MLflow on DagShub\nExperiment Tracking]
    F --> G{val_diffuser_loss\nimproved?}
    G -- Yes --> H[Download Best Model]
    G -- No --> I[Skip downstream stages]
    H --> H2[Evaluate Model\nGenerate Samples]
    H2 --> J[ONNX Export\nCLIP + VAE + UNet]
    J --> J2[TensorRT Compile\nGPU-optimized engines]
    J2 --> K[Push to HF Hub]
    K --> L[HF Spaces\nDDIM · CPU Inference]
```

### DVC Stages

| # | Stage | What it does |
|---|---|---|
| 1 | `caption-generator` | Auto-generates detailed captions for ~8K flower images using Florence-2-large |
| 2 | `precompute-embeddings` | Pre-caches CLIP text embeddings to disk, eliminating per-batch CLIP inference |
| 3 | `data-push` | Uploads dataset (images + captions + embeddings) to S3 for SageMaker |
| 4 | `training` | Launches distributed DeepSpeed training on SageMaker (2 nodes) |
| 5 | `log_training_model` | Downloads best model from S3 — **only if val loss improved** |
| 6 | `evaluate` | Generates sample images and computes evaluation metrics |
| 7 | `onnx_convert` | Exports CLIP + VAE + UNet to ONNX for CPU inference |
| 8 | `tensorrt_convert` | Compiles ONNX models to TensorRT engines for GPU inference |
| 9 | `push_to_hub` | Pushes all model artifacts to Hugging Face Hub |

Stages 5–9 are automatically skipped by DVC when the model doesn't improve, preventing unnecessary exports and deployments.

### CI/CD: Auto-deploy on `git push`

```mermaid
graph LR
    A[git push main] --> B[GitHub Actions]
    B --> C[Upload app.py +\nDockerfile]
    C --> D[HF Spaces\nStreamlit App]
    D --> E[Downloads ONNX\nfrom HF Hub]
    E --> F[DDIM inference\non CPU]
```

---

## Model Architecture

```mermaid
graph LR
    P[Text Prompt] --> CLIP[CLIP ViT-L/14\nfrozen]
    CLIP --> E[Embeddings\n77 × 768]

    I[Training Image] --> VAE_E[VAE Encoder\nSD ft-mse]
    VAE_E --> L[Latent\n16×16×4]

    L --> NOISE[Add Noise\nDDPM · T=1000]
    NOISE --> UNET[Cross-Attn UNet2D\nNoise Predictor]
    E --> UNET
    UNET --> DENOISE[Denoise\nDDIM · 30-50 steps]
    DENOISE --> VAE_D[VAE Decoder]
    VAE_D --> OUT[Generated Image\n128×128]
```

| Component | Details |
|---|---|
| **Text Encoder** | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) — frozen, 768-dim embeddings |
| **VAE** | Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`) — 8× latent compression |
| **UNet** | `UNet2DConditionModel` with cross-attention — channels [192, 384, 576] |
| **Scheduler** | DDPM (1000 steps, training) → DDIM (30–50 steps, inference) |
| **Output** | 128 × 128 px |

---

## Training

| | |
|---|---|
| **Dataset** | Oxford 102 Flowers (~8,189 images) |
| **Captions** | Auto-generated with Florence-2-large |
| **Platform** | AWS SageMaker — 2× `ml.g4dn.xlarge` (NVIDIA T4, 16 GB each) |
| **Distribution** | DeepSpeed ZeRO Stage 2 — optimizer states + gradients partitioned |
| **Precision** | FP16 mixed precision |
| **Optimizer** | AdamW (lr=2e-4, weight_decay=1e-2) + cosine warmup |
| **Batch size** | 64 |
| **Epochs** | 75 |
| **Gradient clipping** | 0.5 |
| **CFG training** | 10% unconditional dropout |
| **Best val loss** | 0.3124 |
| **Experiment logs** | [DagShub MLflow](https://dagshub.com/aniketpoojari/Text-To-Image-Diffusion.mlflow) |

---

## Tech Stack

| Component | Technology |
|---|---|
| Caption generation | Florence-2-large (Microsoft) |
| Text encoding | CLIP ViT-L/14 (OpenAI) |
| Image compression | Stable Diffusion VAE ft-mse |
| Noise prediction | UNet2DConditionModel (HF Diffusers) |
| Training infrastructure | AWS SageMaker (multi-node spot instances) |
| Distributed training | DeepSpeed ZeRO Stage 2 |
| Experiment tracking | MLflow on DagShub |
| Pipeline orchestration | DVC |
| Model optimization | ONNX Runtime (CPU) · TensorRT (GPU) |
| Deployment | Hugging Face Spaces + GitHub Actions CI/CD |
| App framework | Streamlit |

---

## Project Structure

```
├── .github/workflows/
│   └── deploy_to_hf_spaces.yml     # CI/CD — auto-deploy to HF Spaces on push
├── data/raw/flowers/
│   ├── images/                      # Oxford 102 Flowers dataset (~8K images)
│   ├── captions/                    # Florence-2-generated captions
│   └── embeddings/                  # Pre-cached CLIP embeddings
├── samples/                         # Generated sample images
├── saved_models/
│   ├── app.py                       # Streamlit app (local + HF Spaces)
│   ├── Dockerfile                   # HF Spaces container
│   ├── requirements.txt             # HF Spaces dependencies
│   ├── diffuser.pth                 # Trained UNet weights (DVC-tracked)
│   ├── onnx_models/                 # ONNX exports (DVC-tracked)
│   └── trt_models/                  # TensorRT engines (DVC-tracked)
├── src/
│   ├── code/                        # SageMaker training container
│   │   ├── dataloader.py            # Dataset with CLIP embedding cache
│   │   └── training_sagemaker_deepspeed.py
│   ├── caption_generator.py         # Florence-2 caption pipeline
│   ├── precompute_embeddings.py     # CLIP embedding precomputation
│   ├── trainingjob.py               # SageMaker job launcher
│   ├── log_training_model.py        # Best model download (conditional)
│   ├── evaluate.py                  # Model evaluation + sample generation
│   ├── onnx_converter.py            # PyTorch → ONNX export
│   ├── tensorrt_converter.py        # ONNX → TensorRT compilation
│   ├── push_to_hub.py               # Upload to HF Hub
│   ├── upload.py                    # Dataset upload to S3
│   └── common.py                    # Shared utilities
├── notebooks/
│   └── Diffusion.ipynb              # EDA & exploration
├── dvc.yaml                         # Pipeline definition (9 stages)
├── params.yaml.template             # Config template
├── requirements.txt                 # Project dependencies
├── MODEL_CARD.md                    # Model documentation
└── SYSTEM_DESIGN.md                 # System architecture & design decisions
```

---

## Setup & Usage

### 1. Clone and install

```bash
git clone https://github.com/aniketpoojari/Text-To-Image-Diffusion.git
cd Text-To-Image-Diffusion
pip install -r requirements.txt
```

### 2. Configure

```bash
cp params.yaml.template params.yaml
# Fill in AWS credentials, MLflow URI, and HF token
```

### 3. Run the full pipeline

```bash
dvc repro
```

### 4. Run the app locally

```bash
cd saved_models
streamlit run app.py
```

Supports three inference backends — select in the sidebar:
- **ONNX Runtime** — CPU-optimized (default on HF Spaces)
- **TensorRT** — fastest on NVIDIA GPUs
- **PyTorch** — full precision, any device

### 5. Deploy to HF Spaces

Automated via GitHub Actions on every push to `main`. One-time setup:

1. Create a Docker Space on [huggingface.co/new-space](https://huggingface.co/new-space)
2. Add GitHub repo secrets: `HF_TOKEN`

---

## Key Design Decisions

**Why precompute CLIP embeddings?**
CLIP inference was the per-batch bottleneck during training. Pre-caching embeddings to disk as `.pt` files eliminates this overhead entirely and enables multi-worker data loading.

**Why DeepSpeed ZeRO Stage 2?**
Partitions optimizer states and gradients across GPUs. This fits the full UNet + pretrained VAE on 2× T4 instances (16 GB each) that would otherwise OOM with standard data parallelism.

**Why conditional model download?**
The `log_training_model` stage compares the new run's `val_diffuser_loss` against the current model's metadata. If no improvement, file hashes stay the same and DVC automatically skips all downstream stages — no unnecessary ONNX exports or HF Hub uploads.

**Why DDIM over DDPM at inference?**
DDPM requires 1000 denoising steps. DDIM achieves comparable quality in 30–50 steps — a 20–33× speedup critical for CPU-based deployment on HF Spaces.

**Why ONNX for deployment?**
ONNX Runtime CPU inference is significantly faster than PyTorch CPU due to graph optimization and kernel fusion. This makes the HF Spaces demo (CPU-only free tier) practical.

For a deeper dive into architecture and trade-offs, see **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**.

# Model Card: Text-to-Image Diffusion Model

## Model Details

| | |
|---|---|
| **Architecture** | Cross-Attentional UNet2DConditionModel (Hugging Face Diffusers) |
| **Text Encoder** | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) — frozen |
| **VAE** | Stable Diffusion VAE ft-mse (`stabilityai/sd-vae-ft-mse`) — fine-tuned |
| **Noise Scheduler** | DDPM (training) · DDIM (inference) |
| **Image Resolution** | 128 × 128 px |
| **Latent Space** | 16 × 16 × 4 |

## Training

| | |
|---|---|
| **Dataset** | Oxford 102 Flowers |
| **Captions** | Auto-generated with Florence-2-large (`microsoft/Florence-2-large`) |
| **Training Platform** | AWS SageMaker — 2× `ml.g4dn.xlarge` (NVIDIA T4 16 GB each) |
| **Distribution** | DeepSpeed ZeRO Stage 2 |
| **Precision** | FP16 mixed precision |
| **Optimizer** | AdamW with cosine LR warmup (DeepSpeed WarmupLR) |
| **Gradient Clipping** | 0.5 |
| **CFG Training** | 10% unconditional dropout for classifier-free guidance |

## Inference

- **Scheduler**: DDIM — 30 steps recommended (fast CPU inference)
- **CFG Scale**: 7.5 (recommended)
- **Inference time**: ~2–4 min on CPU (HF Spaces), ~5–10 s on NVIDIA GPU

## Evaluation

| Metric | Value |
|---|---|
| Train diffuser loss | *see MLflow run* |
| Val diffuser loss | *see MLflow run* |

Experiment tracking: [DagShub MLflow](https://dagshub.com/aniketpoojari/Text-To-Image-Diffusion.mlflow)

## Limitations

- Domain-specific: trained only on flower images — prompts outside this domain will produce poor results.
- Low resolution: 128 × 128 px. Not suitable for high-fidelity generation.
- Limited dataset: ~8 K images. Generation quality improves significantly with more data.

## Intended Use

- Portfolio demonstration of an end-to-end diffusion model training pipeline on AWS.
- Educational reference for distributed training with DeepSpeed on SageMaker.
- Not intended for production use.

## Ethical Considerations

- Dataset contains only flower photographs — no personal data.
- Model is unlikely to generate harmful or biased content given its narrow training domain.

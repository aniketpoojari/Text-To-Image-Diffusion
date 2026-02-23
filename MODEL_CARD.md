# Model Card: Text-to-Image Diffusion Model

## Model Details

| | |
|---|---|
| **Architecture** | Cross-Attentional UNet2DConditionModel (Hugging Face Diffusers) |
| **Text Encoder** | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) — frozen |
| **VAE** | Stable Diffusion VAE ft-mse (`stabilityai/sd-vae-ft-mse`) — frozen |
| **Noise Scheduler** | DDPM (training, 1000 steps) · DDIM (inference, 30–50 steps) |
| **Image Resolution** | 128 × 128 px |
| **Latent Space** | 16 × 16 × 4 |

## Training

| | |
|---|---|
| **Dataset** | Oxford 102 Flowers (~8,189 images, 102 categories) |
| **Captions** | Auto-generated with Florence-2-large (`microsoft/Florence-2-large`) |
| **Training Platform** | AWS SageMaker — 2× `ml.g4dn.xlarge` (NVIDIA T4 16 GB each) |
| **Distribution** | DeepSpeed ZeRO Stage 2 |
| **Precision** | FP16 mixed precision |
| **Optimizer** | AdamW (lr=2e-4, weight_decay=1e-2) with cosine LR warmup |
| **Batch Size** | 64 |
| **Epochs** | 75 |
| **Gradient Clipping** | 0.5 |
| **CFG Training** | 10% unconditional dropout for classifier-free guidance |

## Evaluation

| Metric | Value |
|---|---|
| Best val diffuser loss | 0.3124 |
| Avg generation time (GPU) | 2.37 s/image (CUDA, DDIM 50 steps, CFG 5.0) |
| Avg generation time (CPU) | ~2–4 min/image (ONNX Runtime, DDIM 30 steps) |

Experiment tracking: [DagShub MLflow](https://dagshub.com/aniketpoojari/Text-To-Image-Diffusion.mlflow)

## Inference

- **Scheduler**: DDIM — 30 steps (CPU) to 50 steps (GPU) recommended
- **CFG Scale**: 5.0–7.5 recommended
- **Backends**: ONNX Runtime (CPU), TensorRT (GPU), PyTorch (any)

## Limitations

- **Domain-specific**: trained only on flower images — prompts outside this domain will produce poor results.
- **Low resolution**: 128 × 128 px. Not suitable for high-fidelity generation.
- **Limited dataset**: ~8K images. Generation quality would improve significantly with more data.

## Intended Use

- Portfolio demonstration of an end-to-end diffusion model training pipeline.
- Educational reference for distributed training with DeepSpeed on SageMaker.
- Not intended for production use.

## Ethical Considerations

- Dataset contains only flower photographs — no personal data.
- Model is unlikely to generate harmful or biased content given its narrow training domain.

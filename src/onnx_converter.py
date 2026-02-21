"""
Convert trained PyTorch models to ONNX for optimized CPU inference.

Exports:
  - CLIP text encoder  →  saved_models/onnx_models/clip_text_encoder.onnx
  - VAE decoder        →  saved_models/onnx_models/vae_decoder.onnx
  - UNet (diffuser)    →  saved_models/onnx_models/unet.onnx

Run from the project root:
    python src/onnx_converter.py
"""

import os
import warnings
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Exporting aten::index operator of advanced indexing")

DIFFUSER_PATH = "saved_models/diffuser.pth"
ONNX_DIR = "saved_models/onnx_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")


# ── 1. CLIP Text Encoder ──────────────────────────────────────────────────────
print("\nExporting CLIP text encoder...")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

dummy_input = tokenizer(
    "This is a dummy prompt",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77,
)
input_ids = dummy_input["input_ids"].to(DEVICE)
attention_mask = dummy_input["attention_mask"].to(DEVICE)

torch.onnx.export(
    clip_model,
    (input_ids, attention_mask),
    os.path.join(ONNX_DIR, "clip_text_encoder.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print("  CLIP text encoder exported.")
del clip_model


# ── 2. VAE Decoder ────────────────────────────────────────────────────────────
print("Exporting VAE decoder...")


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample


vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
vae_decoder = VAEDecoderWrapper(vae)

dummy_latents = torch.randn(1, 4, 32, 32).to(DEVICE)  # 256px / 8 (VAE compression) = 32

torch.onnx.export(
    vae_decoder,
    dummy_latents,
    os.path.join(ONNX_DIR, "vae_decoder.onnx"),
    input_names=["latents"],
    output_names=["image"],
    opset_version=17,
    do_constant_folding=True,
)
print("  VAE decoder exported.")
del vae, vae_decoder


# ── 3. UNet (Diffuser) ────────────────────────────────────────────────────────
print("Exporting UNet diffuser...")


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latents, timesteps, encoder_hidden_states):
        return self.unet(latents, timesteps, encoder_hidden_states).sample


diffuser = torch.load(DIFFUSER_PATH, map_location=DEVICE, weights_only=False).eval()
unet_wrapper = UNetWrapper(diffuser)

# Batch size 2 for classifier-free guidance (uncond + cond concatenated)
dummy_latents = torch.randn(2, 4, 32, 32).to(DEVICE)  # 256px / 8 = 32
dummy_timestep = torch.tensor([50, 50], dtype=torch.float32).to(DEVICE)
dummy_text_emb = torch.randn(2, 77, 768).to(DEVICE)

torch.onnx.export(
    unet_wrapper,
    (dummy_latents, dummy_timestep, dummy_text_emb),
    os.path.join(ONNX_DIR, "unet.onnx"),
    input_names=["latents", "timesteps", "encoder_hidden_states"],
    output_names=["noise_pred"],
    dynamic_axes={
        "latents": {0: "batch_size"},
        "timesteps": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
        "noise_pred": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print("  UNet exported.")
del diffuser, unet_wrapper

print(f"\nAll models exported to {ONNX_DIR}/")
print("  clip_text_encoder.onnx")
print("  vae_decoder.onnx")
print("  unet.onnx")

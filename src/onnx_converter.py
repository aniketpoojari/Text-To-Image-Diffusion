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
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from common import read_params

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")
warnings.filterwarnings("ignore", message="Exporting aten::index operator of advanced indexing")

def convert_to_onnx(config_path):
    config = read_params(config_path)
    
    # Pull settings from config
    vae_h, vae_w = map(int, config["vae"]["image_size"].split(","))
    latent_h, latent_w = vae_h // 8, vae_w // 8
    max_length = config["clip"]["max_length"]
    diffuser_path = config["log_trained_model"]["diffuser_dir"]
    onnx_dir = "saved_models/onnx_models"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(onnx_dir, exist_ok=True)
    print(f"Using device: {device}")

    # ── 1. CLIP Text Encoder ──────────────────────────────────────────────────
    print("\nExporting CLIP text encoder...")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    dummy_input = tokenizer(
        "This is a dummy prompt",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = dummy_input["input_ids"].to(device)
    attention_mask = dummy_input["attention_mask"].to(device)

    torch.onnx.export(
        clip_model,
        (input_ids, attention_mask),
        os.path.join(onnx_dir, "clip_text_encoder.onnx"),
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

    # ── 2. VAE Decoder ────────────────────────────────────────────────────────
    print("Exporting VAE decoder...")

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, latents):
            return self.vae.decode(latents).sample

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    vae_decoder = VAEDecoderWrapper(vae)

    dummy_latents = torch.randn(1, 4, latent_h, latent_w).to(device)

    torch.onnx.export(
        vae_decoder,
        dummy_latents,
        os.path.join(onnx_dir, "vae_decoder.onnx"),
        input_names=["latents"],
        output_names=["image"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "latents": {0: "batch_size", 2: "height", 3: "width"},
            "image": {0: "batch_size", 2: "out_height", 3: "out_width"},
        }
    )
    print("  VAE decoder exported.")
    del vae, vae_decoder

    # ── 3. UNet (Diffuser) ────────────────────────────────────────────────────
    print("Exporting UNet diffuser...")

    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        def forward(self, latents, timesteps, encoder_hidden_states):
            return self.unet(latents, timesteps, encoder_hidden_states).sample

    if not os.path.exists(diffuser_path):
        raise FileNotFoundError(f"Model not found at {diffuser_path}. Train and log the model first.")
        
    diffuser = torch.load(diffuser_path, map_location=device, weights_only=False).eval()
    unet_wrapper = UNetWrapper(diffuser)

    # Batch size 2 for classifier-free guidance
    dummy_latents = torch.randn(2, 4, latent_h, latent_w).to(device)
    dummy_timestep = torch.tensor([50, 50], dtype=torch.long).to(device)
    dummy_text_emb = torch.randn(2, max_length, 768).to(device)

    torch.onnx.export(
        unet_wrapper,
        (dummy_latents, dummy_timestep, dummy_text_emb),
        os.path.join(onnx_dir, "unet.onnx"),
        input_names=["latents", "timesteps", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "latents": {0: "batch_size", 2: "height", 3: "width"},
            "timesteps": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "noise_pred": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("  UNet exported.")
    del diffuser, unet_wrapper

    print(f"\nAll models exported to {onnx_dir}/")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    convert_to_onnx(config_path=parsed_args.config)

"""
Model evaluation: generate sample images from test prompts.

Loads the saved diffuser.pth, generates 5 images with DDIM (50 steps, CFG 7.5),
saves them to samples/, and writes evaluation_results.json.

Usage:
    python src/evaluate.py --config=params.yaml

DVC stage: runs after log_training_model, before push_to_hub.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from common import read_params

TEST_PROMPTS = [
    "A bright yellow sunflower with green leaves in bright sunlight",
    "A red rose with water droplets on its petals",
    "A purple lavender field at sunset with warm golden light",
    "A white daisy on a grassy meadow with soft bokeh background",
    "A pink cherry blossom tree in spring with blue sky",
]

DDIM_STEPS = 50
CFG_SCALE = 7.5
LATENT_H, LATENT_W = 32, 32  # 256px / 8 (VAE compression factor)
MAX_LENGTH = 77


def generate_image(prompt, unet, vae, tokenizer, text_encoder, scheduler, device):
    tokens = tokenizer(
        prompt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_LENGTH,
    ).to(device)
    null_tokens = tokenizer(
        "", return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        cond = text_encoder(**tokens).last_hidden_state
        uncond = text_encoder(**null_tokens).last_hidden_state

    latents = torch.randn((1, 4, LATENT_H, LATENT_W), device=device)
    scheduler.set_timesteps(DDIM_STEPS)
    latents = latents * scheduler.init_noise_sigma

    with torch.no_grad():
        for t in scheduler.timesteps:
            lat_in = torch.cat([latents, latents])
            txt = torch.cat([uncond, cond])
            t_batch = torch.full((2,), t.item(), dtype=torch.long, device=device)
            noise_pred_raw = unet(
                lat_in, t_batch, encoder_hidden_states=txt, return_dict=False
            )[0]
            n_uncond, n_cond = noise_pred_raw.chunk(2)
            noise_pred = n_uncond + CFG_SCALE * (n_cond - n_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        decoded = vae.decode(latents / 0.18215).sample
        # VAE outputs [-1,1] → convert to [0,1] → uint8
        img = decoded[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
        img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)

    return Image.fromarray(img)


def evaluate(config_path):
    config = read_params(config_path)
    diffuser_path = config["log_trained_model"]["diffuser_dir"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading models on {device}...")
    unet = torch.load(diffuser_path, map_location=device, weights_only=False).eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    os.makedirs("samples", exist_ok=True)
    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"[{i+1}/{len(TEST_PROMPTS)}] {prompt}")
        t0 = time.time()
        img = generate_image(prompt, unet, vae, tokenizer, text_encoder, scheduler, device)
        elapsed = time.time() - t0

        out_path = f"samples/sample_{i+1}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}  ({elapsed:.1f}s on {device.upper()})")
        results.append({"prompt": prompt, "image": out_path, "generation_time_s": round(elapsed, 2)})

    avg_time = sum(r["generation_time_s"] for r in results) / len(results)
    summary = {
        "device": device,
        "ddim_steps": DDIM_STEPS,
        "cfg_scale": CFG_SCALE,
        "avg_generation_time_s": round(avg_time, 2),
        "samples": results,
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Avg: {avg_time:.1f}s/image on {device.upper()}")
    print("Commit samples/*.png to git to display them in the README.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    evaluate(config_path=parsed_args.config)

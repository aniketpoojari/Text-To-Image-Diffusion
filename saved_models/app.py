import torch
import numpy as np
import streamlit as st
import onnxruntime as ort
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

## General Model Setup

# Inference Function
def generate_image_general(prompt, config, tokenizer, text_encoder, vae, unet, noise_scheduler):

    # Step 1: Tokenize and encode text
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=config['max_length']).to(config['device'])
    with torch.no_grad():
        text_embeddings = text_encoder(**tokens).last_hidden_state

    # Step 2: Generate random noise in latent space
    noise_shape = (1, 3, config['size'][0], config['size'][1])
    noise = torch.randn(noise_shape).to(config['device'])

    with torch.no_grad():
        with torch.autocast(device_type=config['device'], dtype=torch.float16):
            # Step 3: Encode noise to latent space
            latents = vae.encode(noise).latent_dist.sample() * 0.18215
                
            # Step 4: Denoise using the UNet
            for t in range(config['T']-1, 0, -1):
                # Get the predicted noise from the U-Net
                noise_t = unet(latents, torch.tensor([t]).to(config['device']), encoder_hidden_states=text_embeddings, return_dict=False)[0]
                latents = noise_scheduler.step(noise_t, t, latents).prev_sample


            # Step 5: Decode latents to image
            latents = 1 / 0.18215 * latents  # Scale back latents
            image = vae.decode(latents).sample

    # Step 6: Convert to PIL image and return
    # image = (image[0] / 2 + 0.5).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    image = image[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return image

# Load General Model
def load_general_model(config):
    vae = torch.load("vae.pth", map_location=torch.device(config['device']), weights_only=False).eval()
    unet = torch.load("diffuser.pth", map_location=torch.device(config['device']), weights_only=False).eval()
    model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(config['device'])
    noise_scheduler = DDPMScheduler(num_train_timesteps=config['T'], beta_start=1e-4, beta_end=0.02)
    return vae, unet, tokenizer, text_encoder, noise_scheduler


## ONNX Model Setup

# Inference Function
def generate_image_onnx(prompt, config, tokenizer, clip_session, unet_session, vae_session, noise_scheduler):
    # === 1. Text → Text Embedding ===
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", truncation=True, max_length=config["max_length"])
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    text_embeddings = clip_session.run(["last_hidden_state"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })[0]  # shape: (1, 77, 768)

    # === 2. Random Latents ===
    latent_shape = (1, 4, config["size"][0] // 8, config["size"][1] // 8)
    latents = np.random.randn(*latent_shape).astype(np.float32) * 0.18215

    # === 3. Denoising Loop ===
    for t in noise_scheduler.timesteps:
        timestep = np.array([t], dtype=np.float32)

        noise_pred = unet_session.run(["noise_pred"], {
            "latents": latents,
            "timesteps": timestep,
            "encoder_hidden_states": text_embeddings.astype(np.float32),
        })[0]

        latents = noise_scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

    # === 4. Decode with VAE ===
    latents = latents / 0.18215
    decoded_image = vae_session.run(["image"], {"latents": latents.astype(np.float32)})[0]

    # === 5. Convert to (H, W, C) format for display ===
    image = decoded_image[0].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image = np.clip(image, 0, 1)  # Clamp to [0, 1] for display

    return image

# Load ONNX Model
def load_onnx_model(config):
    """
    Loads ONNX Runtime inference sessions, tokenizer, and noise scheduler.
    Returns all components as a dictionary.
    """

    providers = ["CUDAExecutionProvider"] if config["device"] == "cuda" else ["CPUExecutionProvider"]

    sessions = {
        "vae_session": ort.InferenceSession("onnx_models/vae_decoder.onnx", providers=providers),
        "unet_session": ort.InferenceSession("onnx_models/unet.onnx", providers=providers),
        "clip_session": ort.InferenceSession("onnx_models/clip_text_encoder.onnx", providers=providers),
    }

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    scheduler = DDPMScheduler(
        num_train_timesteps=config["T"],
        beta_start=1e-4,
        beta_end=0.02,
    )
    scheduler.set_timesteps(config["T"])

    return {
        **sessions,
        "tokenizer": tokenizer,
        "noise_scheduler": scheduler,
    }

config = {
            "size": (128, 128),
            "max_length": 77,
            "T": 1000,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

method = "on"

with st.spinner("Loading models..."):
    if method == "onnx":
        models = load_onnx_model(config)
    else:
        vae, unet, tokenizer, text_encoder, noise_scheduler = load_general_model(config)

# Title of the app
st.title("Text to Image Generator")

user_input = st.text_input("Enter some text:")

if st.button("Start Prediction"):

    if user_input:

        # start timer
        import time
        start_time = time.time()

        if method == "onnx":
            generated_image = generate_image_onnx(user_input, config, models["tokenizer"], models["clip_session"], models["unet_session"], models["vae_session"], models["noise_scheduler"])
        else:
            generated_image = generate_image_general(user_input, config, tokenizer, text_encoder, vae, unet, noise_scheduler)
        
        # end timer
        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Execution time: {execution_time:.2f} seconds")

        st.image(generated_image, width=400)
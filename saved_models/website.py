import streamlit as st
import torch
from torch.nn import functional as F
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from PIL import Image
import numpy as np

# Inference Function
def generate_image(prompt, config, tokenizer, text_encoder, vae, unet, noise_scheduler):

    # Step 1: Tokenize and encode text
    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=config['max_length']).to(config['device'])
    with torch.no_grad():
        text_embeddings = text_encoder(**tokens).last_hidden_state

    # Step 2: Generate random noise in latent space
    noise_shape = (1, 3, config['size'][0], config['size'][1])
    noise = torch.randn(noise_shape).to(config['device'])

    # Step 3: Encode noise to latent space
    with torch.no_grad():
        latents = vae.encode(noise).latent_dist.sample() * 0.18215
                
    # Step 4: Denoise using the UNet
    with torch.no_grad():
        for t in range(config['T']-1, 0, -1):
            # Get the predicted noise from the U-Net
            noise_t = unet(latents, torch.tensor([t]).to(config['device']), encoder_hidden_states=text_embeddings, return_dict=False)[0]
            latents = noise_scheduler.step(noise_t, t, latents).prev_sample


    # Step 5: Decode latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents  # Scale back latents
        image = vae.decode(latents).sample

    # Step 6: Convert to PIL image and return
    image = (image[0] / 2 + 0.5).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return image

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = torch.load("../saved_models/vae.pth").to(device)
    unet = torch.load("../saved_models/diffuser.pth").to(device)
    vae.eval()
    unet.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)

    return vae, unet, tokenizer, text_encoder, noise_scheduler

vae, unet, tokenizer, text_encoder, noise_scheduler = load_model()

# Title of the app
st.title("Text to Image Generator")

user_input = st.text_input("Enter some text:")

if st.button("Start Prediction"):

    if user_input:
        config = {
            "size": (128, 128),
            "max_length": 77,
            "T": 1000,
            "device": "cuda"
        }
        generated_image = generate_image(user_input, config, tokenizer, text_encoder, vae, unet, noise_scheduler)


        st.image(generated_image, width=400)
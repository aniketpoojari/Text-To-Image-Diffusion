import torch
import streamlit as st
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

# Inference Function
def generate_image(prompt, config, tokenizer, text_encoder, vae, unet, noise_scheduler):

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

@st.cache_resource
def load_model(config):
    vae = torch.load("vae.pth", map_location=torch.device(config['device'])).eval()
    unet = torch.load("diffuser.pth", map_location=torch.device(config['device'])).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(config['device'])
    noise_scheduler = DDPMScheduler(num_train_timesteps=config['T'], beta_start=1e-4, beta_end=0.02)
    return vae, unet, tokenizer, text_encoder, noise_scheduler


torch.set_float32_matmul_precision('medium')

config = {
            "size": (128, 128),
            "max_length": 77,
            "T": 1000,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

vae, unet, tokenizer, text_encoder, noise_scheduler = load_model(config)

# Title of the app
st.title("Text to Image Generator")

user_input = st.text_input("Enter some text:")

if st.button("Start Prediction"):

    if user_input:

        # start timer
        import time
        start_time = time.time()
        
        generated_image = generate_image(user_input, config, tokenizer, text_encoder, vae, unet, noise_scheduler)

        # end timer
        end_time = time.time()
        execution_time = end_time - start_time
        st.write(f"Execution time: {execution_time:.2f} seconds")

        st.image(generated_image, width=400)
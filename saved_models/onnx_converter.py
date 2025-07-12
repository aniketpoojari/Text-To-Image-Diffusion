import torch
from transformers import CLIPTextModel, CLIPTokenizer
import os

os.makedirs("onnx_models", exist_ok=True)

device = "cuda"

# === 1. Export CLIPTextModel ===
model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPTextModel.from_pretrained(model_name).to(device).eval()
tokenizer = CLIPTokenizer.from_pretrained(model_name)

dummy_input = tokenizer("This is a dummy prompt", return_tensors="pt", padding="max_length", truncation=True, max_length=77)
input_ids = dummy_input["input_ids"].to(device)
attention_mask = dummy_input["attention_mask"].to(device)

torch.onnx.export(
    clip_model,
    (input_ids, attention_mask),
    "onnx_models/clip_text_encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "last_hidden_state": {0: "batch_size"}},
    opset_version=17,
    do_constant_folding=True
)

print("✅ CLIPTextModel exported.")


# === 2. Export VAE decoder ===
class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample

# vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
vae = torch.load("vae.pth", weights_only=False).to(device).eval()
vae_decoder = VAEDecoderWrapper(vae)

dummy_latents = torch.randn(1, 4, 32, 32).to(device)
torch.onnx.export(
    vae_decoder,
    dummy_latents,
    "onnx_models/vae_decoder.onnx",
    input_names=["latents"],
    output_names=["image"],
    opset_version=17,
    do_constant_folding=True
)

print("✅ VAE decoder exported.")


# === 3. Export UNet ===
# You need to configure your UNet with the same parameters as your trained model
diffuser = torch.load("diffuser.pth", weights_only=False).to(device).eval()

class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latents, timesteps, encoder_hidden_states):
        return self.unet(latents, timesteps, encoder_hidden_states).sample

unet_wrapper = UNetWrapper(diffuser)

dummy_latents = torch.randn(1, 4, 32, 32).to(device)
dummy_timestep = torch.tensor([50], dtype=torch.float32).to(device)
dummy_text_emb = torch.randn(1, 77, 768).to(device)

torch.onnx.export(
    unet_wrapper,
    (dummy_latents, dummy_timestep, dummy_text_emb),
    "onnx_models/unet.onnx",
    input_names=["latents", "timesteps", "encoder_hidden_states"],
    output_names=["noise_pred"],
    dynamic_axes={"latents": {0: "batch_size"}, "encoder_hidden_states": {0: "batch_size"}, "noise_pred": {0: "batch_size"}},
    opset_version=17,
    do_constant_folding=True
)

print("✅ UNet exported.")

"""
Text-to-Image Generator — Streamlit App

Inference modes:
  - ONNX      (default on HF Spaces, CPU-optimised)
  - TensorRT  (fastest on local GPU, requires NVIDIA GPU + TensorRT SDK)
  - PyTorch   (full precision, works on any device)

Model source:
  - If HF_MODEL_REPO env var is set  → downloads models from Hugging Face Hub (HF Spaces)
  - Otherwise                         → loads models from local saved_models/ directory

Usage (local):
    cd saved_models && streamlit run app.py

Environment variable for HF Spaces:
    HF_MODEL_REPO=your-username/text-to-image-diffusion
"""

import os
import time

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")
LATENT_H, LATENT_W = 32, 32   # 256px image / 8 (VAE compression)
MAX_LENGTH = 77
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRT_DIR = "trt_models"
_TRT_FILES = ("clip_text_encoder.engine", "unet.engine", "vae_decoder.engine")
TRT_AVAILABLE = (
    DEVICE == "cuda"
    and not HF_MODEL_REPO
    and all(os.path.exists(os.path.join(TRT_DIR, f)) for f in _TRT_FILES)
)


# ── TensorRT engine wrapper ───────────────────────────────────────────────────

class TRTEngine:
    """
    Drop-in replacement for ort.InferenceSession with the same .run() interface.
    Uses PyTorch CUDA tensors — no pycuda required.
    Requires TensorRT >= 8.6 and an NVIDIA GPU.
    """

    def __init__(self, engine_path):
        import tensorrt as trt
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def run(self, output_names, input_feed):
        """
        Mimics ort.InferenceSession.run(output_names, input_feed).
        input_feed: dict of {name: numpy_array}
        Returns: list of numpy arrays in output_names order.
        """
        trt = self._trt
        stream = torch.cuda.current_stream().cuda_stream

        # Upload inputs to GPU and set shapes + addresses
        input_tensors = {}
        for name, arr in input_feed.items():
            t = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
            input_tensors[name] = t
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # Allocate output tensors on GPU
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
        }
        output_tensors = {}
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = dtype_map.get(self.engine.get_tensor_dtype(name), torch.float32)
            t = torch.empty(shape, dtype=dtype, device="cuda")
            output_tensors[name] = t
            self.context.set_tensor_address(name, t.data_ptr())

        self.context.execute_async_v3(stream)
        torch.cuda.synchronize()

        return [output_tensors[name].float().cpu().numpy() for name in output_names]


# ── Model loading helpers ─────────────────────────────────────────────────────

def _get_onnx_path(filename):
    """Return path to an ONNX model, downloading from HF Hub if needed."""
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename=f"onnx/{filename}")
    return os.path.join("onnx_models", filename)


def _get_pth_path():
    """Return path to diffuser.pth, downloading from HF Hub if needed."""
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_MODEL_REPO, filename="diffuser.pth")
    return "diffuser.pth"


def _ort_session(path):
    """Create an ONNX Runtime session with GPU if available, else CPU."""
    if DEVICE == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 ** 3,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=providers)


@st.cache_resource
def load_models(method):
    """Load models (cached across Streamlit reruns)."""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # DDIM scheduler — fast inference (30 steps vs 1000 for DDPM)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    if method == "tensorrt":
        clip_eng = TRTEngine(os.path.join(TRT_DIR, "clip_text_encoder.engine"))
        unet_eng = TRTEngine(os.path.join(TRT_DIR, "unet.engine"))
        vae_eng  = TRTEngine(os.path.join(TRT_DIR, "vae_decoder.engine"))
        # Same tuple layout as onnx: (clip, unet, vae, tokenizer, scheduler)
        return clip_eng, unet_eng, vae_eng, tokenizer, scheduler

    elif method == "onnx":
        clip_sess = _ort_session(_get_onnx_path("clip_text_encoder.onnx"))
        unet_sess = _ort_session(_get_onnx_path("unet.onnx"))
        vae_sess  = _ort_session(_get_onnx_path("vae_decoder.onnx"))
        return clip_sess, unet_sess, vae_sess, tokenizer, scheduler

    else:  # pytorch
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE).eval()
        unet = torch.load(_get_pth_path(), map_location=DEVICE, weights_only=False).eval()
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        # Tuple layout: (vae, unet, tokenizer, text_encoder, scheduler)
        return vae, unet, tokenizer, text_encoder, scheduler


# ── Text embeddings ───────────────────────────────────────────────────────────

def get_text_embeddings(prompt, tokenizer, method, models):
    if method in ("onnx", "tensorrt"):
        clip_sess = models[0]
        inputs = tokenizer(
            prompt, return_tensors="np", padding="max_length",
            truncation=True, max_length=MAX_LENGTH,
        )
        null_inputs = tokenizer(
            "", return_tensors="np", padding="max_length",
            truncation=True, max_length=MAX_LENGTH,
        )
        cond = clip_sess.run(
            ["last_hidden_state"],
            {"input_ids": inputs["input_ids"].astype(np.int64),
             "attention_mask": inputs["attention_mask"].astype(np.int64)},
        )[0]
        uncond = clip_sess.run(
            ["last_hidden_state"],
            {"input_ids": null_inputs["input_ids"].astype(np.int64),
             "attention_mask": null_inputs["attention_mask"].astype(np.int64)},
        )[0]
        return torch.from_numpy(uncond).to(DEVICE), torch.from_numpy(cond).to(DEVICE)

    else:  # pytorch
        _, _, tokenizer, text_encoder, _ = models
        device = torch.device(DEVICE)
        toks = tokenizer(prompt, return_tensors="pt", padding="max_length",
                         truncation=True, max_length=MAX_LENGTH).to(device)
        null_toks = tokenizer("", return_tensors="pt", padding="max_length",
                              truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            cond = text_encoder(**toks).last_hidden_state
            uncond = text_encoder(**null_toks).last_hidden_state
        return uncond, cond


# ── Preview helper ────────────────────────────────────────────────────────────

def decode_preview(latents, method, models):
    """Decode latents to a displayable numpy image without storing state."""
    preview = latents / 0.18215
    try:
        if method in ("onnx", "tensorrt"):
            vae_sess = models[2]
            img_np = vae_sess.run(["image"], {"latents": preview.cpu().numpy()})[0]
            # VAE decoder outputs [-1, 1] — convert to [0, 1]
            img = (np.clip(img_np[0].transpose(1, 2, 0), -1, 1) + 1.0) / 2.0
        else:
            vae = models[0]
            with torch.no_grad():
                decoded = vae.decode(preview).sample
            img = decoded[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
            img = (img + 1.0) / 2.0
        return img
    except Exception:
        return None


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_image(prompt, cfg_scale, steps, models, method):
    device = torch.device(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    uncond_emb, cond_emb = get_text_embeddings(prompt, models[3], method, models)

    # Initialise latents
    latents = torch.randn((1, 4, LATENT_H, LATENT_W), device=device)
    scheduler = models[-1]
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma

    progress_bar = st.progress(0)
    status = st.empty()
    preview_placeholder = st.empty()

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            lat_in = torch.cat([latents, latents])
            txt_emb = torch.cat([uncond_emb, cond_emb])
            t_batch = torch.full((2,), t.item(), dtype=torch.float32, device=device)

            if method in ("onnx", "tensorrt"):
                unet_sess = models[1]
                noise_np = unet_sess.run(
                    ["noise_pred"],
                    {
                        "latents": lat_in.cpu().numpy(),
                        "timesteps": t_batch.cpu().numpy(),
                        "encoder_hidden_states": txt_emb.cpu().numpy(),
                    },
                )[0]
                noise_pred_raw = torch.from_numpy(noise_np).to(device)
            else:
                unet = models[1]
                noise_pred_raw = unet(
                    lat_in, t_batch.long(), encoder_hidden_states=txt_emb, return_dict=False
                )[0]

            n_uncond, n_cond = noise_pred_raw.chunk(2)
            noise_pred = n_uncond + cfg_scale * (n_cond - n_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            progress_bar.progress((i + 1) / steps)
            status.write(f"Step {i + 1}/{steps}")

            # Show a preview every 10 steps
            if (i + 1) % 10 == 0 or i == steps - 1:
                img = decode_preview(latents.clone(), method, models)
                if img is not None:
                    preview_placeholder.image(img, caption=f"Step {i + 1}/{steps}", width=400)

    progress_bar.empty()
    status.empty()

    final = decode_preview(latents, method, models)
    return final


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Text-to-Image Generator", layout="centered")
st.title("Text-to-Image Generation")
st.caption(
    "Diffusion model trained on AWS SageMaker with DeepSpeed ZeRO Stage 2. "
    + ("Running on HF Spaces (CPU)." if HF_MODEL_REPO else f"Running locally on {DEVICE.upper()}.")
)

with st.sidebar:
    st.header("Settings")

    # Build backend options based on what's available
    if HF_MODEL_REPO:
        backend_options = ["onnx"]
    elif TRT_AVAILABLE:
        backend_options = ["tensorrt", "onnx", "pytorch"]
    elif DEVICE == "cuda":
        backend_options = ["onnx", "pytorch"]
    else:
        backend_options = ["onnx", "pytorch"]

    method = st.selectbox(
        "Inference Backend",
        backend_options,
        index=0,
        help=(
            "TensorRT: fastest on local GPU. "
            "ONNX: fast on CPU (used on HF Spaces). "
            "PyTorch: full precision, any device."
        ),
    )
    cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 7.5, 0.5,
                          help="Higher = more prompt-adherent, lower = more creative.")
    steps = st.slider("DDIM Steps", 10, 100, 30, 5,
                      help="30 steps gives good quality. More steps = slower.")
    st.info(f"Using DDIM scheduler — {steps} steps on {DEVICE.upper()}")

with st.spinner("Loading models (first run may take a minute)..."):
    models = load_models(method)

prompt = st.text_input("Prompt", placeholder="A red rose with water droplets on its petals")

if st.button("Generate", type="primary") and prompt.strip():
    t0 = time.time()
    img = generate_image(prompt, cfg_scale, steps, models, method)
    elapsed = time.time() - t0

    if img is not None:
        st.success(f"Generated in {elapsed:.1f}s  |  {method.upper()} · DDIM {steps} steps · CFG {cfg_scale}")
        st.image(img, caption=prompt, width=400)
    else:
        st.error("Generation failed. Check model files and try again.")

"""
Build TensorRT engines from ONNX models for optimized local GPU inference.

Inputs:  saved_models/onnx_models/{clip_text_encoder,unet,vae_decoder}.onnx
Outputs: saved_models/trt_models/{clip_text_encoder,unet,vae_decoder}.engine

Engines are GPU-architecture specific — rebuild if you change hardware.
FP16 precision is used when the GPU supports it (all NVIDIA Turing/Ampere+).

Run from the project root:
    python src/tensorrt_converter.py

DVC stage: runs after onnx_convert. Requires NVIDIA GPU + TensorRT >= 8.6.
"""

import os
import sys

try:
    import tensorrt as trt
except ModuleNotFoundError:
    print("TensorRT is not installed — skipping engine build.")
    print("Install with: pip install tensorrt")
    print("Engines require NVIDIA GPU + matching CUDA toolkit.")
    os.makedirs("saved_models/trt_models", exist_ok=True)
    open("saved_models/trt_models/unavailable.txt", "w").close()
    sys.exit(0)

ONNX_DIR = "saved_models/onnx_models"
TRT_DIR = "saved_models/trt_models"
os.makedirs(TRT_DIR, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=True, profiles=None):
    """Parse an ONNX file and serialize a TensorRT engine to disk."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled.")

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse {onnx_path}")

    if profiles:
        for profile_spec in profiles:
            profile = builder.create_optimization_profile()
            for inp_name, (min_shape, opt_shape, max_shape) in profile_spec.items():
                profile.set_shape(inp_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

    print("  Building engine (may take several minutes on first run)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"  Saved: {engine_path}  ({size_mb:.1f} MB)")


# ── 1. CLIP Text Encoder ──────────────────────────────────────────────────────
# Dynamic batch: CFG inference always sends batch=2 (uncond + cond)
print("\nBuilding CLIP text encoder engine...")
build_engine(
    onnx_path=os.path.join(ONNX_DIR, "clip_text_encoder.onnx"),
    engine_path=os.path.join(TRT_DIR, "clip_text_encoder.engine"),
    fp16=True,
    profiles=[{
        "input_ids":      ([1, 77], [2, 77], [2, 77]),
        "attention_mask": ([1, 77], [2, 77], [2, 77]),
    }],
)

# ── 2. UNet ───────────────────────────────────────────────────────────────────
# Dynamic batch: batch=2 for CFG (uncond + cond concatenated)
# Latent spatial: 16×16 (128px image ÷ 8 VAE compression)
print("\nBuilding UNet engine...")
build_engine(
    onnx_path=os.path.join(ONNX_DIR, "unet.onnx"),
    engine_path=os.path.join(TRT_DIR, "unet.engine"),
    fp16=True,
    profiles=[{
        "latents":               ([1, 4, 32, 32], [2, 4, 32, 32], [2, 4, 32, 32]),
        "timesteps":             ([1],             [2],             [2]),
        "encoder_hidden_states": ([1, 77, 768],    [2, 77, 768],    [2, 77, 768]),
    }],
)

# ── 3. VAE Decoder ────────────────────────────────────────────────────────────
# Fixed batch=1 — no dynamic axes were set during ONNX export
print("\nBuilding VAE decoder engine...")
build_engine(
    onnx_path=os.path.join(ONNX_DIR, "vae_decoder.onnx"),
    engine_path=os.path.join(TRT_DIR, "vae_decoder.engine"),
    fp16=True,
    profiles=None,
)

print(f"\nAll TensorRT engines built in {TRT_DIR}/")
print("  clip_text_encoder.engine")
print("  unet.engine")
print("  vae_decoder.engine")

import os
import sys
import argparse
from common import read_params

try:
    import tensorrt as trt
except ModuleNotFoundError:
    print("TensorRT is not installed — skipping engine build.")
    os.makedirs("saved_models/trt_models", exist_ok=True)
    open("saved_models/trt_models/unavailable.txt", "w").close()
    sys.exit(0)

def build_trt_engines(config_path):
    config = read_params(config_path)
    
    # Pull settings
    vae_h, vae_w = map(int, config["vae"]["image_size"].split(","))
    lh, lw = vae_h // 8, vae_w // 8
    max_length = config["clip"]["max_length"]
    
    onnx_dir = "saved_models/onnx_models"
    trt_dir = "saved_models/trt_models"
    os.makedirs(trt_dir, exist_ok=True)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def build_engine(onnx_path, engine_path, fp16=True, profiles=None):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        cfg = builder.create_builder_config()
        if fp16 and builder.platform_has_fast_fp16:
            cfg.set_flag(trt.BuilderFlag.FP16)
        
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError(f"Failed to parse {onnx_path}")

        if profiles:
            for profile_spec in profiles:
                profile = builder.create_optimization_profile()
                for inp_name, (min_s, opt_s, max_s) in profile_spec.items():
                    profile.set_shape(inp_name, min_s, opt_s, max_s)
                cfg.add_optimization_profile(profile)

        print(f"  Building {engine_path}...")
        serialized = builder.build_serialized_network(network, cfg)
        if serialized is None:
            raise RuntimeError(f"Failed to build engine for {onnx_path}. Check TRT logs above.")
        with open(engine_path, "wb") as f:
            f.write(serialized)

    # 1. CLIP
    print("\nBuilding CLIP engine...")
    build_engine(
        os.path.join(onnx_dir, "clip_text_encoder.onnx"),
        os.path.join(trt_dir, "clip_text_encoder.engine"),
        profiles=[{"input_ids": ([1, max_length], [2, max_length], [2, max_length]),
                   "attention_mask": ([1, max_length], [2, max_length], [2, max_length])}]
    )

    # 2. UNet
    print("\nBuilding UNet engine...")
    build_engine(
        os.path.join(onnx_dir, "unet.onnx"),
        os.path.join(trt_dir, "unet.engine"),
        profiles=[{"latents": ([1, 4, lh, lw], [2, 4, lh, lw], [2, 4, lh, lw]),
                   "timesteps": ([1], [2], [2]),
                   "encoder_hidden_states": ([1, max_length, 768], [2, max_length, 768], [2, max_length, 768])}]
    )

    # 3. VAE
    print("\nBuilding VAE engine...")
    build_engine(
        os.path.join(onnx_dir, "vae_decoder.onnx"),
        os.path.join(trt_dir, "vae_decoder.engine"),
        profiles=[{"latents": ([1, 4, lh, lw], [1, 4, lh, lw], [1, 4, lh, lw])}]
    )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    build_trt_engines(config_path=parsed_args.config)

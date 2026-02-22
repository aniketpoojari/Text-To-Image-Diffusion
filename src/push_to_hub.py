"""
Push trained models to Hugging Face Hub.

Uploads to the model repo specified in params.yaml (huggingface.model_repo):
  - saved_models/diffuser.pth                        →  diffuser.pth
  - saved_models/onnx_models/clip_text_encoder.onnx  →  onnx/clip_text_encoder.onnx
  - saved_models/onnx_models/vae_decoder.onnx        →  onnx/vae_decoder.onnx
  - saved_models/onnx_models/unet.onnx               →  onnx/unet.onnx

The repository is auto-created if it does not exist.

Usage:
    python src/push_to_hub.py --config=params.yaml
"""

import argparse
import os
from datetime import datetime

from common import read_params
from huggingface_hub import HfApi, create_repo


ONNX_FILES = [
    ("saved_models/onnx_models/clip_text_encoder.onnx", "onnx/clip_text_encoder.onnx"),
    ("saved_models/onnx_models/vae_decoder.onnx", "onnx/vae_decoder.onnx"),
    ("saved_models/onnx_models/unet.onnx", "onnx/unet.onnx"),
]
DIFFUSER_FILE = ("saved_models/diffuser.pth", "diffuser.pth")


def push_to_hub(config_path):
    config = read_params(config_path)

    repo_id = config["huggingface"]["model_repo"]
    # Use HF_TOKEN env var if set, otherwise None (uses cached token from login)
    hf_token = os.environ.get("HF_TOKEN")

    api = HfApi(token=hf_token)

    # Auto-create the model repo if it doesn't exist
    create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    files_to_upload = [DIFFUSER_FILE] + ONNX_FILES

    for local_path, repo_path in files_to_upload:
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Expected file not found: {local_path}\n"
                "Run onnx_convert stage first: dvc repro onnx_convert"
            )
        print(f"Uploading {local_path} → {repo_id}/{repo_path} ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )

    print(f"\nAll files pushed to https://huggingface.co/{repo_id}")

    with open("push_completion.txt", "w") as f:
        f.write(f"Pushed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Repo: https://huggingface.co/{repo_id}\n")
        f.write(f"Files: {[r for _, r in files_to_upload]}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    push_to_hub(config_path=parsed_args.config)

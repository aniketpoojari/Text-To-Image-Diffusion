"""
Precompute CLIP text embeddings for all flower images.

Runs locally (once) before uploading data to S3. The embeddings are packed into
flowers.zip alongside images and captions, so SageMaker training containers never
need to load the CLIP model — they just read the .pt files from disk.

Outputs to data/raw/flowers/embeddings/:
  {image_name}.pt       — per-image embedding, shape [77, 768], float32
  null_embedding.pt     — empty-prompt embedding for CFG unconditional path

Usage:
    python src/precompute_embeddings.py --config=params.yaml

DVC stage: runs after caption-generator, before data-push.
"""

import argparse
import os

import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from common import read_params


def precompute_embeddings(config_path: str):
    config = read_params(config_path)

    data_raw   = config["data"]["raw"]                  # e.g. data/raw/flowers
    max_length = config["clip"]["max_length"]            # 77

    images_dir     = os.path.join(data_raw, "images")
    captions_dir   = os.path.join(data_raw, "captions")
    embeddings_dir = os.path.join(data_raw, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")

    model_name   = "openai/clip-vit-large-patch14"
    tokenizer    = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device).eval()

    image_files = sorted(os.listdir(images_dir))

    # Determine which embeddings are still missing
    missing = [
        f for f in image_files
        if not os.path.exists(
            os.path.join(embeddings_dir, os.path.splitext(f)[0] + ".pt")
        )
    ]

    null_emb_path = os.path.join(embeddings_dir, "null_embedding.pt")

    if not missing and os.path.exists(null_emb_path):
        print(f"All {len(image_files)} embeddings already cached. Nothing to do.")
        return

    print(f"Computing embeddings for {len(missing)} images...")

    with torch.no_grad():

        # Null embedding (unconditional path for CFG)
        if not os.path.exists(null_emb_path):
            null_tok = tokenizer(
                "", padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            null_emb = text_encoder(
                input_ids=null_tok.input_ids.to(device),
                attention_mask=null_tok.attention_mask.to(device),
            ).last_hidden_state.squeeze().cpu()
            torch.save(null_emb, null_emb_path)
            print(f"  Saved null_embedding.pt  shape={tuple(null_emb.shape)}")

        # Per-image embeddings
        for image_file in tqdm(missing, desc="Embedding images"):
            base_name    = os.path.splitext(image_file)[0]
            emb_path     = os.path.join(embeddings_dir, base_name + ".pt")
            caption_file = base_name + ".txt"
            caption_path = os.path.join(captions_dir, caption_file)

            with open(caption_path, "r", encoding="utf-8") as f:
                text = f.readline().strip()

            tokens = tokenizer(
                text, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            emb = text_encoder(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state.squeeze().cpu()
            torch.save(emb, emb_path)

    del text_encoder
    torch.cuda.empty_cache()

    total = len(os.listdir(embeddings_dir))
    print(f"\nDone. {total} files in {embeddings_dir}")
    print("These will be packed into flowers.zip by the data-push stage.")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    precompute_embeddings(config_path=parsed_args.config)

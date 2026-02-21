"""
Pack the flowers dataset (images + captions + precomputed embeddings) into a zip
and upload it to the S3 bucket used by SageMaker for training data.

By including pre-computed CLIP embeddings in the zip, SageMaker training containers
never need to load the CLIP model — they read .pt files directly from disk.

Zip layout (mirrors data/raw/flowers/):
    flowers/
        images/          ← raw JPEG images
        captions/        ← Florence-2 generated captions (.txt)
        embeddings/      ← CLIP embeddings (.pt) + null_embedding.pt

Usage:
    python src/upload.py --config=params.yaml

DVC stage: runs after precompute-embeddings.
"""

import argparse
import os
import zipfile
from datetime import datetime

import boto3

from common import read_params


def create_zip(source_dir: str, zip_path: str):
    """
    Zip the entire source_dir directory.
    Paths inside the zip are relative to source_dir's parent so the zip contains
    a top-level 'flowers/' folder matching the structure the SageMaker container expects.
    """
    parent = os.path.dirname(source_dir)
    folder_name = os.path.basename(source_dir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file in sorted(files):
                abs_path = os.path.join(root, file)
                # arcname keeps 'flowers/images/xxx.jpg', 'flowers/embeddings/xxx.pt', etc.
                arc_name = os.path.join(
                    folder_name, os.path.relpath(abs_path, source_dir)
                )
                zf.write(abs_path, arc_name)

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"Created {zip_path}  ({size_mb:.1f} MB)")


def upload(config_path: str):
    config = read_params(config_path)

    data_raw       = config["data"]["raw"]                             # data/raw/flowers
    zip_path       = data_raw + ".zip"                                 # data/raw/flowers.zip
    zip_file_name  = os.path.basename(zip_path)                        # flowers.zip
    s3_train_data  = config["pytorch_estimator"]["s3_train_data"]      # s3://bucket/flowers.zip
    s3_bucket_name = s3_train_data.split("/")[2]                       # bucket name only

    embeddings_dir = os.path.join(data_raw, "embeddings")
    if not os.path.isdir(embeddings_dir) or not os.listdir(embeddings_dir):
        raise RuntimeError(
            f"Embeddings not found at {embeddings_dir}.\n"
            "Run the precompute-embeddings stage first:  dvc repro precompute-embeddings"
        )

    # Build the zip fresh every time (embeddings may have changed)
    print(f"Packing {data_raw}/ → {zip_path} ...")
    create_zip(source_dir=data_raw, zip_path=zip_path)

    print(f"Uploading {zip_file_name} → s3://{s3_bucket_name}/{zip_file_name} ...")
    s3 = boto3.client("s3")
    s3.upload_file(zip_path, s3_bucket_name, zip_file_name)
    print("Upload complete.")

    with open("upload_completion.txt", "w") as f:
        f.write(f"S3 Upload Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Contents: images/ + captions/ + embeddings/\n")
        f.write(f"Destination: s3://{s3_bucket_name}/{zip_file_name}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    upload(config_path=parsed_args.config)

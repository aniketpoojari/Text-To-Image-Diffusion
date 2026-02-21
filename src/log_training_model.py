from common import read_params
import argparse
import json
import os
from datetime import datetime
import mlflow
import pandas as pd
import boto3
from botocore.exceptions import ClientError


METADATA_PATH = "saved_models/model_metadata.json"


def get_current_best_loss():
    """Return the val_diffuser_loss of the currently saved model, or inf if none."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        return float(metadata.get("val_diffuser_loss", float("inf")))
    return float("inf")


def log_production_model(config_path):
    config = read_params(config_path)

    server_uri = config["mlflow"]["server_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    s3_bucket = config["mlflow"]["s3_mlruns_bucket"]
    diffuser_dest = config["log_trained_model"]["diffuser_dir"]

    mlflow.set_tracking_uri(server_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

    # Find the best finished run sorted by lowest val_diffuser_loss
    runs_df = pd.DataFrame(
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.val_diffuser_loss ASC"],
        )
    )

    if runs_df.empty:
        raise ValueError("No finished runs found in MLflow. Train the model first.")

    # Find the best run that actually has a model uploaded to S3.
    # Old runs that predate the S3-upload step are skipped automatically.
    s3 = boto3.client("s3")
    best_run = best_loss = run_id = None
    for _, row in runs_df.iterrows():
        rid  = row["run_id"]
        loss = float(row.get("metrics.val_diffuser_loss", float("inf")))
        try:
            s3.head_object(Bucket=s3_bucket, Key=f"{rid}/diffuser.pth")
            run_id    = rid
            best_loss = loss
            best_run  = row
            break
        except ClientError:
            print(f"  Skipping run {rid}: no model in S3.")

    if best_run is None:
        raise ValueError("No finished run has a model in S3. Train the model first.")

    current_loss = get_current_best_loss()

    print(f"Best run : {run_id}  (val_diffuser_loss={best_loss:.4f})")
    print(f"Saved    : val_diffuser_loss={current_loss:.4f}")

    if best_loss >= current_loss:
        print("No improvement detected — skipping download. Downstream stages will not re-run.")
        return

    print(f"Improvement found ({current_loss:.4f} → {best_loss:.4f}). Downloading model...")

    os.makedirs(os.path.dirname(diffuser_dest), exist_ok=True)

    s3_key = f"{run_id}/diffuser.pth"
    s3.download_file(s3_bucket, s3_key, diffuser_dest)
    print(f"Downloaded s3://{s3_bucket}/{s3_key} → {diffuser_dest}")

    metadata = {
        "run_id": run_id,
        "val_diffuser_loss": best_loss,
        "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)

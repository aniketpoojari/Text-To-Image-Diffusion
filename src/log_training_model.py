from common import read_params
import argparse
import mlflow
import pandas as pd
import shutil
import boto3


def log_production_model(config_path):
    config = read_params(config_path)

    server_uri = config["mlflow"]["server_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(server_uri)

    # get experiment id
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # get best model
    df = pd.DataFrame(mlflow.search_runs(experiment_ids=experiment_id))
    df = df[df["status"] == "FINISHED"]

    vae = df[df["metrics.val_vae_loss"] == df["metrics.val_vae_loss"].min()]
    vae_src = vae['artifact_uri'].values[0].split("mlflow-diffusion-aniket/")[1] + "/vae/data/model.pth"

    diffuser = df[df["metrics.val_diffuser_loss"] == df["metrics.val_diffuser_loss"].min()]
    diffuser_src = diffuser['artifact_uri'].values[0].split("mlflow-diffusion-aniket/")[1] + "/diffuser/data/model.pth"

    # copy model
    vae_dest = config["log_trained_model"]["vae_dir"]
    diffuser_dest = config["log_trained_model"]["diffuser_dir"]
    # shutil.copyfile(vae_src, vae_dest)
    # shutil.copyfile(diffuser_src, diffuser_dest)

    s3 = boto3.client('s3')

    bucket_name = config["mlflow"]["s3_mlruns_bucket"]
    
    # Download file
    s3.download_file(bucket_name, vae_src, vae_dest)
    s3.download_file(bucket_name, diffuser_src, diffuser_dest)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)

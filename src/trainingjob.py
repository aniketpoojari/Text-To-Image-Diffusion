from common import read_params
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def setup_training(config_path):
    config = read_params(config_path)

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Environment variables for training script inside the container pulled from params.yaml
    environment = {
        "TRAIN_SIZE": str(config['data']['train_size']),
        "VAL_SIZE": str(config['data']['val_size']),
        
        "VAE_IMAGE_SIZE": config['vae']['image_size'],
        "MAX_LENGTH": str(config['clip']['max_length']),
        "BATCH_SIZE": str(config['training']['batch_size']),
        
        "T": str(config['DDPMScheduler']['T']),
        
        "UNET_IMAGE_SIZE": config['unet']['image_size'],
        "IN_CHANNELS": str(config['unet']['in_channels']),
        "OUT_CHANNELS": str(config['unet']['out_channels']),
        "DOWN_BLOCK_TYPES": config['unet']['down_block_types'],
        "UP_BLOCK_TYPES": config['unet']['up_block_types'],
        "MID_BLOCK_TYPE": config['unet']['mid_block_type'],
        "BLOCK_OUT_CHANNELS": config['unet']['block_out_channels'],
        "LAYERS_PER_BLOCK": str(config['unet']['layers_per_block']),
        "NORM_NUM_GROUPS": str(config['unet']['norm_num_groups']),
        "CROSS_ATTENTION_DIM": str(config['unet']['cross_attention_dim']),
        "ATTENTION_HEAD_DIM": str(config['unet']['attention_head_dim']),
        "DROPOUT": str(config['unet']['dropout']),
        "TIME_EMBEDDING_TYPE": config['unet']['time_embedding_type'],
        "ACT_FN": config['unet']['act_fn'],
        
        "UNET_LEARNING_RATE": str(config['training']['unet_learning_rate']),
        "WEIGHT_DECAY": str(config['training']['weight_decay']),
        "NUM_EPOCHS": str(config['training']['num_epochs']),
        "USE_EMA": str(config['training']['use_ema']),

        "EXPERIMENT_NAME": config['mlflow']['experiment_name'],
        "RUN_NAME": config['mlflow']['run_name'],
        "REGISTERED_MODEL_NAME": config['mlflow']['registered_model_name'],
        "SERVER_URI": config['mlflow']['server_uri'],
        "S3_MLRUNS_BUCKET": config['mlflow']['s3_mlruns_bucket'],

        "MLFLOW_TRACKING_USERNAME": config['mlflow']['tracking_username'],
        "MLFLOW_TRACKING_PASSWORD": config['mlflow']['tracking_password'],
    }

    # SageMaker parses these regex patterns from CloudWatch logs
    # and displays them as live charts in the training job console
    metric_definitions = [
        # per-epoch losses
        {"Name": "train:loss",       "Regex": r"Train loss : ([0-9.]+)"},
        {"Name": "val:loss",         "Regex": r"Val loss: ([0-9.]+)"},
        {"Name": "epoch:time_s",     "Regex": r"Epoch time : ([0-9.]+)s"},
        # per-batch metrics
        {"Name": "batch:loss",       "Regex": r"Loss ([0-9.]+) \|"},
        {"Name": "batch:imgs_s",     "Regex": r"([0-9.]+) imgs/s"},
        {"Name": "batch:data_ms",    "Regex": r"data +([0-9.]+)ms  vae"},
        {"Name": "batch:vae_ms",     "Regex": r"vae +([0-9.]+)ms  unet"},
        {"Name": "batch:unet_ms",    "Regex": r"unet +([0-9.]+)ms  bwd"},
        {"Name": "batch:bwd_ms",     "Regex": r"bwd +([0-9.]+)ms  batch"},
        {"Name": "batch:total_ms",   "Regex": r"batch +([0-9.]+)ms \|"},
        {"Name": "batch:overall_pct","Regex": r"Overall +([0-9.]+)%"},
        # avg timings from epoch summary
        {"Name": "avg:data_ms",      "Regex": r"Avg/batch.*data ([0-9.]+)ms"},
        {"Name": "avg:vae_ms",       "Regex": r"Avg/batch.*vae ([0-9.]+)ms"},
        {"Name": "avg:unet_ms",      "Regex": r"Avg/batch.*unet ([0-9.]+)ms"},
        {"Name": "avg:bwd_ms",       "Regex": r"Avg/batch.*bwd ([0-9.]+)ms"},
        {"Name": "avg:total_ms",     "Regex": r"Avg/batch.*total ([0-9.]+)ms"},
        # GPU memory
        {"Name": "gpu:allocated_gb", "Regex": r"Allocated: ([0-9.]+)GB"},
        {"Name": "gpu:cached_gb",    "Regex": r"Cached: ([0-9.]+)GB"},
        # early stopping
        {"Name": "best_val",         "Regex": r"best_val: ([0-9.]+)"},
        {"Name": "patience",         "Regex": r"patience: ([0-9]+)/"},
    ]

    # Configure PyTorch estimator
    use_spot = config['pytorch_estimator']['use_spot_instances']
    estimator_kwargs = dict(
        entry_point=config['pytorch_estimator']['entry_point'],
        source_dir=config['pytorch_estimator']['source_dir'],
        role=config['pytorch_estimator']['role'],
        framework_version=config['pytorch_estimator']['framework_version'],
        py_version=config['pytorch_estimator']['py_version'],
        instance_count=config['pytorch_estimator']['instance_count'],
        instance_type=config['pytorch_estimator']['instance_type'],
        use_spot_instances=use_spot,
        max_run=config['pytorch_estimator']['max_run'],
        environment=environment,
        metric_definitions=metric_definitions,
        distribution={
            "deepspeed": {
                "enabled": True
            }
        },
    )
    if use_spot:
        estimator_kwargs["max_wait"] = config['pytorch_estimator']['max_wait']

    estimator = PyTorch(**estimator_kwargs
    )

    # Define data channels
    data = {
        'train': config['pytorch_estimator']['s3_train_data'],
    }
    
    # Start training with less verbose logs
    estimator.fit(inputs=data, logs="minimal")  # 'minimal' reduces log output

    # update the training completion time so that DVC can use that file to check whether the training step is complete
    with open("training_completion.txt", "w") as file:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        file.write("Training Completed at " + formatted_datetime)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    setup_training(config_path=parsed_args.config)
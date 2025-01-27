from common import read_params
import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime


def setup_training(config_path):
    config = read_params(config_path)

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Environment variables for training
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
        
        "VAE_LEARNING_RATE": config['training']['vae_learning_rate'],
        "UNET_LEARNING_RATE": config['training']['unet_learning_rate'],
        "WEIGHT_DECAY": config['training']['weight_decay'],
        "NUM_EPOCHS": str(config['training']['num_epochs']),

        "EXPERIMENT_NAME": config['mlflow']['experiment_name'],
        "RUN_NAME": config['mlflow']['run_name'],
        "REGISTERED_MODEL_NAME": config['mlflow']['registered_model_name'],
        "SERVER_URI": config['mlflow']['server_uri'],
        "S3_MLRUNS_BUCKET": config['mlflow']['s3_mlruns_bucket'],

        "MLFLOW_TRACKING_USERNAME": config['mlflow']['tracking_username'],
        "MLFLOW_TRACKING_PASSWORD": config['mlflow']['tracking_password'],
    }


    # Configure PyTorch estimator
    estimator = PyTorch(
        entry_point=config['pytorch_estimator']['entry_point'],
        source_dir=config['pytorch_estimator']['source_dir'],
        role=config['pytorch_estimator']['role'],
        framework_version=config['pytorch_estimator']['framework_version'],
        py_version=config['pytorch_estimator']['py_version'],
        instance_count=config['pytorch_estimator']['instance_count'],
        instance_type=config['pytorch_estimator']['instance_type'],
        use_spot_instances=config['pytorch_estimator']['use_spot_instances'],
        max_wait=config['pytorch_estimator']['max_wait'],
        max_run=config['pytorch_estimator']['max_run'],
        environment=environment,
        distribution={
            "pytorchddp": {
                "enabled": True,
                "processes_per_host": 1
            }
        }
    )

    # Define data channels
    data = {
        'train': config['pytorch_estimator']['s3_train_data'],
    }
    
    # Start training
    estimator.fit(inputs=data)

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
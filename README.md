# 🚀 **Text-to-Image Generation with Diffusion Models on AWS SageMaker**

## 📚 **Table of Contents**
- [📄 Introduction](#introduction)
- [📂 DVC Pipeline (`dvc.yaml`)](#dvc-pipeline-dvcyaml)
- [🔧 Parameters (`params.yaml`)](#parameters-paramsyaml)
- [⚡️ SageMaker Trigger (`trainingjob.py`)](#sagemaker-trigger-trainingjobpy)
- [📝 Training Code (`code` Folder)](#training-code-code-folder)
- [📦 Log Training Model (`log_training_model.py`)](#log-training-model-log_training_modelpy)
- [📥 Download and Setup](#download-and-setup)
- [🚀 Run DVC Pipeline](#run-dvc-pipeline)
- [📈 Model Evaluation](#model-evaluation)
- [🎉 Results](#results)
- [🚀 Usage](#usage)


## 📄 **Introduction**
This project generates high-quality images from text captions using a **diffusion model** trained on AWS SageMaker. The main components include:

- **CLIP:** Converts text captions into embeddings to guide image generation.  
- **VAE (Variational Autoencoder):** Compresses images into a smaller latent space and reconstructs them efficiently.  
- **Cross-Attentional UNet:** Predicts noise at each step to gradually refine the generated image.  
- **AWS SageMaker:** Handles large-scale distributed training in the cloud.  
- **MLflow on Dagshub:** Manages model tracking, logging, and versioning.  


## 📂 **DVC Pipeline (`dvc.yaml`)**
- Defines the entire project pipeline.  
- Controls the workflow for training, evaluation, and deployment.  
- You can update `dvc.yaml` to modify the pipeline or change training configurations.


## 🔧 **Parameters (`params.yaml`)**
- configures the training process.
- You can update `params.yaml` to change the training hyperparameters.


## ⚡️ **SageMaker Trigger (`trainingjob.py`)**
- Handles the training process on AWS SageMaker.
- Loads the variables from `params.yaml`.
- Uploads the `code` folder to AWS SageMaker.
- Provisions the necessary compute instances on AWS SageMaker.
- loads the sagemaker environment with the necessary variables for the training process.
- Handles the input and output S3 buckets. 


## 📝 **Training Code (`code` Folder)**
- The `code` folder contains the training code for running the model on AWS SageMaker.  
- `training_sagemaker.py` handles the training process.
    - Loads the variables from the environment variables.
    - Does the distributed training.
    - logs the meta data to MLflow on Dagshub.
    - saves the best model to S3 bucket.


## 📦 **Log Training Model (`log_training_model.py`)**
- Downloads the best model from the **S3 bucket** after training.  
- Uses MLflow logs from Dagshub to identify the best-performing model.  
- Saves the model locally for further evaluation or deployment.  


## 📥 **Download and Setup**
1. **Clone the Repository:**  
```bash
git clone https://github.com/aniketpoojari/Text-to-Image-Generation.git
```
2. **Install Dependencies:**  
```bash
pip install -r requirements.txt
```


## 🔄 **Run DVC Pipeline**
- Run the DVC pipeline to start the training process.  
```bash
dvc repro
```
💡 *Note: Each AWS instance costs approximately $0.80 per hour. I provisioned 2 instances for 40 minutes.*


## 📈 **Model Evaluation**
1. **VAE Evaluation:**  
   - **MSE Loss:** Evaluates how accurately the VAE reconstructs images.  
2. **Diffusion Model Evaluation:**  
   - **MSE Loss:** Measures how well the diffusion model predicts noise and improves generated images.  


## 🎉 **Results**
- **VAE:** Low reconstruction loss due to using a pretrained VAE from Hugging Face.  
- **Diffusion Model:** Low noise prediction loss, demonstrating effective text-to-image generation.  


## 🚀 **Usage**
1. **Run the Best Model:**  
```bash
# Navigate to the saved models directory
cd saved_models
# Launch the web interface
streamlit run website.py
```
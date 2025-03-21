# 🚀 Project Name
**Text-to-Image Generation Diffusion Model using Huggingface and Distibuted AWS SageMaker Training Jobs**

## 📜 Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data](#data)
5. [Model Training](#model-training)
6. [SageMaker Integration](#sagemaker-integration)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Usage](#usage)

## 📄 Introduction
This project focuses on text-to-image generation using diffusion models. The goal is to generate high-quality images conditioned on text captions. The project leverages a Variational Autoencoder (VAE) and a cross-attetional UNet-based diffusion model to achieve this. Key components include:

- **CLIP (Contrastive Language-Image Pre-training)**: CLIP is used to encode text captions into meaningful embeddings, guiding the diffusion model to generate images that are semantically aligned with the input text. CLIP ensures high-quality, contextually accurate image generation.
- **VAE**: An autoencoder that encodes images into latent representations and decodes them back into images, this helps in reducing the size of the image fed into the unet.
- **Cross-Attetional Unet-based Diffusion Models**: A diffusion-based generative model is trained to predict the noise added to images from text captions at each timestep. The cross-attention-unet takes the text embedded captions, latent space representation of noise image from VAE and time steps as inputs. It then predicts the noise added to the image at each timestep.
- **AWS SageMaker Training Jobs Integration**: The project utilizes AWS sagemaker traning jobs to train the model on cloud in a distributed manner.
- **MLflow Tracking**: Experiment tracking, hyperparameter logging, and model versioning are managed using MLflow in Dagshub Cloud.

## 🌟 Features
1. [Local Training](src/training.py) - Trains the model locally, update the dvc.yaml if you want to train locally
2. [Cloud Training](src/codes) - This folder contains the code for training the model on AWS SageMaker.
3. [Sagemaker Trigger](src/trainingjob.py) - Triggers the training job on AWS SageMaker, uploads the code folder to the Sagemaker instance
4. [Log Training Model](src/log_training_model.py) - Gets the details of the best model from Dagshub mlflow and downloads the model from S3 bucket.
5. [dvc.yaml](dvc.yaml) - Defines the pipelines for the project.
6. [params.yaml](params.yaml) - Defines the hyperparameters for the project.

## 🚚 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aniketpoojari/Text-to-Image-Generation.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data
- **Image-Text Dataset**: Oxford 102 Flowers dataset of flower images with captions.

## 🤖 Model Training
    - Save data to S3 bucket.
    - Update params.yaml to update hyperparameters and other configurations.
    - Provision AWS SageMaker instances.
   ```bash
    dvc repo
   ```
   "Each instance costs approximately $0.80 per hour. I provisioned two instances, and they ran for 40 minutes."

## 📈 Evaluation
- **VAE**: Evaluated using **MSE loss** to measure how well the VAE can reconstruct images from latent representations.
- **Diffusion Model**: Evaluated using **MSE loss** to measure how well the model can denoise images.

## 🎉 Results
- **VAE**: Since we have used pretrained VAE model from huggingface we have a low reconstruction loss.
- **Diffusion Model**: Achieved low noise prediction loss, indicating effective image generation conditioned on text captions.

## 🚀 Usage
   ```bash
   # Where the best models are saved after training
   cd saved_models
   streamlit run website.py
   ```
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from typing import Tuple
import os
import random

class TextImageDataLoader(Dataset):
    def __init__(self, datadir, image_size: Tuple[int, int], max_text_length: int):
        super().__init__()

        self.datalist = os.listdir(datadir + '/images')
        # random.seed(42)
        # random.shuffle(images_list)
        self.image_size = image_size
        self.max_text_length = max_text_length

        model_name = "openai/clip-vit-base-patch32"  # Or another CLIP model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to("cuda")

        # Image preprocessing with normalization
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the image and caption filenames
        image_file = self.datalist[idx]
        comment_file = image_file.replace('.jpg', '.txt')  # Assumes consistent naming

        # Load and preprocess the image
        image = Image.open(f'flowers/images/{image_file}').convert("RGB")
        image = self.image_transform(image)

        # Load the text caption
        with open(f'flowers/captions/{comment_file}', 'r') as f:
            text = f.read()

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.to("cuda")
        attention_mask = tokens.attention_mask.to("cuda")

        # Compute text embeddings
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = outputs.last_hidden_state

        return image, text_embeddings.squeeze(), text

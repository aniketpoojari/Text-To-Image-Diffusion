import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Tuple


class TextImageDataLoader(Dataset):
    """
    Dataset for text-image pairs.

    Key design decisions:
    - CLIP embeddings are precomputed once and cached to disk on first run.
      This eliminates per-sample CLIP inference during training (which was the
      main bottleneck) and enables num_workers > 0 in the DataLoader.
    - Images are normalized to [-1, 1] as required by the pretrained VAE
      (stabilityai/sd-vae-ft-mse).
    - range=(start, end): slices the full file list as datalist[start:end].
    """

    def __init__(
        self,
        datadir: str,
        range: Tuple[int, int],
        image_size: Tuple[int, int],
        max_text_length: int,
        training: bool = True,
    ):
        super().__init__()

        self.datadir = datadir
        self.max_text_length = max_text_length
        self.embeddings_dir = os.path.join(datadir, "embeddings")

        # BUG FIX: range is (start, end), not (start, count).
        # Old code: self.datalist[range[0]:range[0]+range[1]] was wrong for val split.
        all_files = sorted(os.listdir(os.path.join(datadir, "images")))
        self.datalist = all_files[range[0]:range[1]]

        # BUG FIX: Normalize images to [-1, 1].
        # The pretrained VAE (stabilityai/sd-vae-ft-mse) was trained on [-1, 1] images.
        # Previously ToTensor() gave [0, 1] and the normalization line was commented out.
        augmentations = [transforms.Resize(image_size)]
        if training:
            augmentations.append(transforms.RandomHorizontalFlip(p=0.5))
        augmentations += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] → [-1,1]
        ]
        self.image_transform = transforms.Compose(augmentations)

        # Precompute and cache all CLIP embeddings (runs once per training job).
        # After this, self.text_encoder is deleted — __getitem__ reads from disk only.
        self._precompute_embeddings()

    # ── Precompute ────────────────────────────────────────────────────────────

    def _precompute_embeddings(self):
        """
        Compute CLIP embeddings for every image in the dataset and save as .pt files.
        Also precomputes and saves the null embedding for CFG dropout.

        On subsequent calls (files already cached), this runs in <1s.
        Enables num_workers > 0 in DataLoader since __getitem__ no longer uses CUDA.
        """
        os.makedirs(self.embeddings_dir, exist_ok=True)
        null_emb_path = os.path.join(self.embeddings_dir, "null_embedding.pt")

        is_dist = torch.distributed.is_initialized()
        # Use local_rank to run once per instance in a distributed job
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Determine which files still need embedding computation
        all_images = sorted(os.listdir(os.path.join(self.datadir, "images")))
        missing = [
            f for f in all_images
            if not os.path.exists(
                os.path.join(self.embeddings_dir, os.path.splitext(f)[0] + ".pt")
            )
        ]

        # Check if local_rank 0 needs to compute
        if local_rank == 0:
            if missing or not os.path.exists(null_emb_path):
                print(f"Precomputing CLIP embeddings for {len(missing)} images...")
                model_name = "openai/clip-vit-large-patch14"
                tokenizer = CLIPTokenizer.from_pretrained(model_name)
                text_encoder = CLIPTextModel.from_pretrained(model_name).to("cuda").eval()

                with torch.no_grad():
                    # Null embedding (for CFG unconditional path)
                    if not os.path.exists(null_emb_path):
                        null_tokens = tokenizer(
                            "", padding="max_length", truncation=True,
                            max_length=self.max_text_length, return_tensors="pt",
                        )
                        null_emb = text_encoder(
                            input_ids=null_tokens.input_ids.to("cuda"),
                            attention_mask=null_tokens.attention_mask.to("cuda"),
                        ).last_hidden_state.squeeze().cpu()
                        torch.save(null_emb, null_emb_path)

                    # Per-image embeddings
                    for image_file in missing:
                        base_name = os.path.splitext(image_file)[0]
                        emb_path = os.path.join(
                            self.embeddings_dir, base_name + ".pt"
                        )
                        caption_path = os.path.join(
                            self.datadir, "captions", base_name + ".txt"
                        )
                        with open(caption_path, "r") as f:
                            text = f.readline().strip()

                        tokens = tokenizer(
                            text, padding="max_length", truncation=True,
                            max_length=self.max_text_length, return_tensors="pt",
                        )
                        emb = text_encoder(
                            input_ids=tokens.input_ids.to("cuda"),
                            attention_mask=tokens.attention_mask.to("cuda"),
                        ).last_hidden_state.squeeze().cpu()
                        torch.save(emb, emb_path)

                # Free GPU memory — CLIP is no longer needed after precomputation
                del text_encoder
                torch.cuda.empty_cache()
                print(f"Embeddings cached to {self.embeddings_dir}")

        # Sync all ranks before loading
        if is_dist:
            torch.distributed.barrier()

        self.null_text_embedding = torch.load(null_emb_path, weights_only=True)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.datalist[idx]
        base_name = os.path.splitext(image_file)[0]

        image = Image.open(
            os.path.join(self.datadir, "images", image_file)
        ).convert("RGB")
        image = self.image_transform(image)  # [-1, 1]

        emb_path = os.path.join(
            self.embeddings_dir, base_name + ".pt"
        )
        embedding = torch.load(emb_path, weights_only=True)

        return image, embedding

    def get_null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return the null (unconditional) embedding expanded to batch_size."""
        return self.null_text_embedding.unsqueeze(0).expand(batch_size, -1, -1)

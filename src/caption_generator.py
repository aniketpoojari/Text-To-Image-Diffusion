from common import read_params
import argparse
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm


def process_batch(image_paths, processor, model, batch_size=4):
    """Process multiple images optimized for RTX 3060"""
    images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        # Resize to exactly 512x512 pixels
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        images.append(image)

    # Use proper prompt for detailed captions
    prompts = ["<DETAILED_CAPTION>"] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=100,
            num_beams=3,  # Optimized for RTX 3060
            do_sample=False
        )

    captions = []
    for i, generated_text in enumerate(processor.batch_decode(generated_ids, skip_special_tokens=True)):  # skip_special_tokens=True removes padding
        parsed_answer = processor.post_process_generation(
            generated_text,
            task="<DETAILED_CAPTION>",
            image_size=(images[i].width, images[i].height)
        )
        caption = parsed_answer["<DETAILED_CAPTION>"]
        # Strip filler prefixes that waste CLIP tokens
        for prefix in ("The image shows ", "The image depicts ", "The image features "):
            if caption.startswith(prefix):
                caption = caption[len(prefix)].upper() + caption[len(prefix)+1:]
                break
        captions.append(caption)

    return captions

def caption_generator(config_path):
    config = read_params(config_path)
    image_directory = os.path.join(config["data"]["raw"], "images")

    print("Loading Florence-2-large model...")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )

    print("Florence-2-large model loaded on GPU")

    # Get all image files from the single image folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(image_directory)
                   if f.lower().endswith(image_extensions)]
    image_paths = [os.path.join(image_directory, f) for f in image_files]

    print(f"Found {len(image_files)} images to process")

    captions_dir = os.path.join(config["data"]["raw"], "captions")
    os.makedirs(captions_dir, exist_ok=True)

    # Check for already processed files
    processed = set()
    if os.path.exists(captions_dir):
        processed = {os.path.splitext(f)[0] for f in os.listdir(captions_dir) if f.endswith('.txt')}

    remaining_files = [(f, p) for f, p in zip(image_files, image_paths)
                       if os.path.splitext(f)[0] not in processed]

    print(f"Found {len(remaining_files)} remaining images to process")

    # Reduced batch size for large model on RTX 3060
    batch_size = 10  # Smaller batch for large model

    remaining_files_only = [item[0] for item in remaining_files]
    remaining_paths_only = [item[1] for item in remaining_files]

    for i in tqdm(range(0, len(remaining_paths_only), batch_size), desc="Processing batches"):
        batch_paths = remaining_paths_only[i:i + batch_size]
        batch_files = remaining_files_only[i:i + batch_size]

        try:
            captions = process_batch(batch_paths, processor, model, len(batch_paths))

            # Save all captions from this batch
            for filename, caption in zip(batch_files, captions):
                base_name = os.path.splitext(filename)[0]
                text_file = os.path.join(captions_dir, f"{base_name}.txt")

                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(caption.strip())

        except torch.cuda.OutOfMemoryError:
            print(f"GPU memory error at batch {i//batch_size + 1}. Processing individually...")
            # Fallback to individual processing
            for path, filename in zip(batch_paths, batch_files):
                try:
                    torch.cuda.empty_cache()  # Clear memory
                    captions = process_batch([path], processor, model, 1)
                    base_name = os.path.splitext(filename)[0]
                    text_file = os.path.join(captions_dir, f"{base_name}.txt")
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(captions[0].strip())
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

        except Exception as e:
            print(f"Error processing batch: {str(e)}")

    print("Processing complete!")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    caption_generator(config_path=parsed_args.config)

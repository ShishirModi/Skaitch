import os
from diffusers import StableDiffusionPipeline
import torch

def download_and_save_model():
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    output_dir = os.path.join(os.path.dirname(__file__), "external", "stable_diffusion")
    
    print(f"Downloading model {model_id}...")
    # This automatically downloads the necessary pieces to HF cache, skipping huge single-file weights
    # We load in fp16 if you want to save space, but let's just use default for CPU compatibility right now
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    
    print(f"Saving explicitly to {output_dir}...")
    pipe.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    download_and_save_model()

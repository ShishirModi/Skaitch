# File: scripts/sd_test.py
# Purpose: quick, minimal Stable Diffusion CPU test (downloads model via diffusers)

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# ===== CONFIG =====
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # mirror of SD v1.5
output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_output.png")

prompt = "close up portrait sketch of Christiano Ronaldo in the center, football player, photorealistic, high detail"
num_inference_steps = 20
guidance_scale = 7.5
height = 512
width = 512
# ==================

def main():
    print(f"Loading model: {model_id} (this will download and cache the model the first time)")
    # If you logged in with huggingface-cli, no token needs to be passed.
    # If not, set HUGGINGFACE_HUB_TOKEN environment variable, or pass use_auth_token="YOUR_TOKEN"
    pipe = StableDiffusionPipeline.from_pretrained("../external/stable_diffusion")
    # Reduce memory usage for CPU
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # Move pipeline to CPU
    pipe = pipe.to("cpu")

    print("Generating image (this can take a few minutes on CPU)...")
    result = pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image = result.images[0]

    print(f"Saving image to: {output_path}")
    image.save(output_path)
    print("Done.")

if __name__ == '__main__':
    main()

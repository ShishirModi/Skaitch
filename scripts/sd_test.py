# File: scripts/sd_test.py
# Purpose: quick Stable Diffusion CLI test with forensic sketch defaults

import sys
import os

# Allow importing prompt_builder from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from prompt_builder import build_forensic_prompt, FORENSIC_DEFAULTS

# ===== CONFIG =====
model_path = os.path.join(os.path.dirname(__file__), "..", "external", "stable_diffusion")
output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_output.png")

# Sample forensic features
sample_features = {
    "Gender": "Male",
    "Age range": "26–35",
    "Face shape": "Oval",
    "Eyes": "Almond",
    "Eyebrows": "Thick",
    "Nose": "Straight",
    "Mouth / Lips": "Thin",
    "Jawline": "Strong",
    "Hair style": "Short cropped",
    "Hair color": "Dark brown",
    "Facial hair": "Stubble",
    "Skin tone": "Medium",
    "Distinguishing marks": "Scar on left cheek",
}

prompt, negative_prompt = build_forensic_prompt(
    sample_features,
    style="Pencil sketch",
    extra_details="",
)

num_inference_steps = FORENSIC_DEFAULTS["num_inference_steps"]
guidance_scale = FORENSIC_DEFAULTS["guidance_scale"]
height = 512
width = 512
# ==================


def main():
    print(f"Loading model from: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe = pipe.to("cpu")

    print(f"Prompt: {prompt}")
    print(f"Negative: {negative_prompt}")
    print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}")
    print("Generating image (this can take a few minutes on CPU)...")

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    image = result.images[0]

    print(f"Saving image to: {output_path}")
    image.save(output_path)
    print("Done.")


if __name__ == "__main__":
    main()
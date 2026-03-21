# File: sketch_refiner.py
# Purpose: SDXL Image-to-Image sketch editing for V2 iterative refinement.
# This module reuses the already-loaded SDXL Base pipeline for zero additional VRAM cost.

import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

def run_sketch_edit(
    pipe,
    sketch_pil: Image.Image,
    mask_pil: Image.Image,
    edit_prompt: str,
    negative_prompt: str,
    strength: float = 0.85, # Increased default to 0.85 for effective structural edits within the mask
    guidance_scale: float = 10.0,
    num_inference_steps: int = 50, # Boosted to guarantee min steps
) -> Image.Image:
    """Apply a targeted edit to an existing sketch using SDXL Regional Inpainting."""
    
    # Fix 1: Manual Component Instantiation to bypass from_pipe hook conflicts
    # We do NOT call enable_model_cpu_offload() here because the underlying
    # PyTorch modules (pipe.unet, pipe.vae, etc.) ALREADY have hooks attached 
    # to them from the parent app.py pipeline.
    i2i_pipe = StableDiffusionXLInpaintPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )

    # Ensure sketch and mask are RGB and at the correct resolution
    sketch_pil = sketch_pil.convert("RGB")
    mask_pil = mask_pil.convert("RGB")

    # Clear any fragmented VRAM before the i2i pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run the inpaint pass
    result = i2i_pipe(
        prompt=edit_prompt,
        image=sketch_pil,
        mask_image=mask_pil,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images[0]

    return result

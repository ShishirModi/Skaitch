# File: sketch_refiner.py
# Purpose: SDXL Image-to-Image sketch editing for V2 iterative refinement.
# This module reuses the already-loaded SDXL Base pipeline for zero additional VRAM cost.

import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from inpaint_enhancements import feather_mask, adaptive_difference_blending

def run_sketch_edit(
    pipe,
    sketch_pil: Image.Image,
    mask_pil: Image.Image,
    edit_prompt: str,
    negative_prompt: str,
    strength: float = 0.85,
    guidance_scale: float = 10.0,
    num_inference_steps: int = 50,
) -> Image.Image:
    """Apply a targeted edit to an existing sketch using SDXL Regional Inpainting.
    
    Integrates inpaint_enhancements.py:
    - Mask feathering: eliminates hard seam artifacts at region boundaries.
    - Adaptive difference blending: prevents cascading edits from drifting 
      away from the original sketch identity.
    """
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

    # Ensure sketch is RGB
    sketch_pil = sketch_pil.convert("RGB")

    # ── Mask feathering: soft edges eliminate visible seams ────────────────────
    # Raw canvas masks have hard 0/255 boundaries → Gaussian blur creates a
    # smooth transition zone so edits blend naturally into surrounding context.
    mask_l = feather_mask(mask_pil.convert("L"), sigma=10)  # keep "L" for blending step
    mask_rgb = mask_l.convert("RGB")  # Diffusers inpaint pipeline requires RGB mask_image

    # Clear any fragmented VRAM before the i2i pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run the inpaint pass
    result = i2i_pipe(
        prompt=edit_prompt,
        image=sketch_pil,
        mask_image=mask_rgb,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images[0]

    # ── Adaptive difference blending: prevents cascading edit drift ────────────
    # Pixels that changed too drastically (hallucinations, global color drift)
    # are conservatively blended back toward the original, preserving evidence
    # integrity across 3+ iterative refinements.
    result = adaptive_difference_blending(
        original=sketch_pil,
        edited=result,
        max_change_factor=0.6,
    )

    return result

# File: sketch_refiner.py
# Purpose: SDXL Image-to-Image sketch editing for V2 iterative refinement.
# This module reuses the already-loaded SDXL Base pipeline for zero additional VRAM cost.

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline


def run_sketch_edit(
    pipe,
    sketch_pil: Image.Image,
    edit_prompt: str,
    negative_prompt: str,
    strength: float = 0.35,
    guidance_scale: float = 10.0,
    num_inference_steps: int = 40,
) -> Image.Image:
    """Apply a targeted edit to an existing sketch using SDXL Image-to-Image.

    This function takes the already-loaded SDXL Base pipeline (text-to-image)
    and converts it to an i2i pipeline using `from_pipe()`. This method
    correctly inherits all memory optimization hooks (CPU offloading, etc.)
    without registering duplicate hooks.

    The 'strength' parameter controls how much the original sketch
    is preserved:
        - 0.2: Very subtle change (slight reshaping)
        - 0.35: Default — noticeable but identity-preserving
        - 0.5: Significant change (e.g., different hairstyle)

    Args:
        pipe: The pre-loaded SDXL text-to-image pipeline from app.py.
              This is converted to i2i mode internally.
        sketch_pil: The current sketch (PIL Image) to edit.
        edit_prompt: The full prompt including both the original features
                     and the edit instruction.
        negative_prompt: Negative prompt for the edit pass.
        strength: Denoise strength (0.0 = no change, 1.0 = full regeneration).
        guidance_scale: How strictly to follow the prompt.
        num_inference_steps: Number of denoising steps.

    Returns:
        The edited sketch as a PIL Image.
    """
    # Convert the text-to-image pipeline to image-to-image mode using from_pipe().
    # This correctly inherits ALL memory optimization hooks (CPU offloading)
    # from the parent pipeline. Do NOT call enable_model_cpu_offload() again —
    # doing so registers duplicate hooks that corrupt inference.
    i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)

    # Ensure sketch is RGB and at the correct resolution
    sketch_pil = sketch_pil.convert("RGB")

    # Clear any fragmented VRAM before the i2i pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run the i2i pass
    result = i2i_pipe(
        prompt=edit_prompt,
        image=sketch_pil,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        output_type="pil",
    ).images[0]

    return result

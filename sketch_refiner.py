# File: sketch_refiner.py
# Purpose: SDXL Image-to-Image sketch editing for V2 iterative refinement.
# This module reuses the already-loaded SDXL Base pipeline for zero additional VRAM cost.
#
# ARCHITECTURE FIX (§2.1): The base SDXL UNet has 4 input channels, but
# StableDiffusionXLInpaintPipeline expects a 9-channel inpaint UNet (4 latent +
# 4 masked-image latent + 1 mask). Using the base UNet in an inpaint pipeline
# caused masks to be silently ignored. The fix uses StableDiffusionXLImg2ImgPipeline
# for the generative pass, then composites the result back onto the original via
# the feathered mask — achieving true regional editing without an inpaint-specific UNet.

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from inpaint_enhancements import (
    feather_mask,
    adaptive_difference_blending,
    prepare_enhanced_inpaint_inputs,
)


def _composite_with_mask(
    original: Image.Image,
    generated: Image.Image,
    mask_l: Image.Image,
) -> Image.Image:
    """Composite generated content onto original using a soft mask.

    Where mask is white (255) the generated content shows through;
    where mask is black (0) the original is preserved. Intermediate
    values produce a smooth blend, eliminating hard seam artifacts.
    """
    orig_np = np.array(original, dtype=np.float32)
    gen_np = np.array(generated.resize(original.size, Image.Resampling.LANCZOS), dtype=np.float32)
    mask_np = np.array(mask_l.resize(original.size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0

    # Expand mask to 3 channels
    if len(orig_np.shape) == 3:
        mask_np = np.stack([mask_np] * 3, axis=-1)

    composited = orig_np * (1.0 - mask_np) + gen_np * mask_np
    return Image.fromarray(np.clip(composited, 0, 255).astype(np.uint8))


def run_sketch_edit(
    pipe,
    sketch_pil: Image.Image,
    mask_pil: Image.Image,
    edit_prompt: str,
    negative_prompt: str,
    edit_instruction: str = "",
    strength: float = 0.85,
    guidance_scale: float = 10.0,
    num_inference_steps: int = 50,
) -> Image.Image:
    """Apply a targeted edit to an existing sketch using SDXL Img2Img + mask compositing.

    Architecture (§2.1 fix): Uses StableDiffusionXLImg2ImgPipeline (which matches
    the base UNet's 4-channel architecture) instead of the inpaint pipeline. The
    mask is enforced via post-generation compositing: generated pixels replace
    originals only inside the feathered mask boundary.

    Integrates inpaint_enhancements.py (§2.4 fix):
    - Full prepare_enhanced_inpaint_inputs() pipeline: feathering, auto-dilation,
      adaptive strength, graduated strength maps.
    - Adaptive difference blending with mask-aware change factor.
    """
    # ── Build Img2Img pipeline from parent components ─────────────────────────
    # We reuse the parent pipeline's modules directly. No enable_model_cpu_offload()
    # call here — the modules already have accelerate hooks from app.py.
    i2i_pipe = StableDiffusionXLImg2ImgPipeline(
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

    # ── Enhanced mask preparation (§2.4 fix) ──────────────────────────────────
    # Uses the full prepare_enhanced_inpaint_inputs() orchestrator instead of
    # calling only feather_mask() — enables auto-dilation, adaptive strength,
    # and graduated strength maps that were previously dead code.
    enhanced = prepare_enhanced_inpaint_inputs(
        original_sketch=sketch_pil,
        user_mask=mask_pil.convert("L"),
        edit_instruction=edit_instruction,
        apply_feathering=True,
        infer_dilation=bool(edit_instruction.strip()),
    )
    mask_feathered = enhanced["mask"]  # feathered + optionally dilated mask (mode "L")
    adaptive_strength = enhanced["strength"]  # mask-size-aware strength recommendation

    # Use the adaptive strength if user hasn't overridden significantly
    effective_strength = strength if abs(strength - 0.85) > 0.01 else adaptive_strength

    # Clear any fragmented VRAM before the i2i pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run the Img2Img generative pass ───────────────────────────────────────
    # Sizing fix: pass explicit width/height so SDXL doesn't default to an
    # internal target resolution that differs from the source image dimensions.
    result = i2i_pipe(
        prompt=edit_prompt,
        image=sketch_pil,
        negative_prompt=negative_prompt,
        strength=effective_strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=sketch_pil.width,
        height=sketch_pil.height,
        output_type="pil",
    ).images[0]

    # ── Mask-based compositing (§2.1 fix) ─────────────────────────────────────
    # Since Img2Img regenerates the full image, we composite using the feathered
    # mask so that only the masked region is replaced and unmasked areas are
    # mathematically preserved from the original.
    result = _composite_with_mask(sketch_pil, result, mask_feathered)

    # ── Adaptive difference blending (§2.3 fix) ──────────────────────────────
    # max_change_factor=0.65: tightened from 0.85 to suppress wild deformations
    # in the composited output. Mask compositing above already anchors unmasked
    # regions to the original, so blending only needs to dampen hallucinated
    # artifacts inside the mask — not preserve deliberate edits against global drift.
    result = adaptive_difference_blending(
        original=sketch_pil,
        edited=result,
        max_change_factor=0.65,
    )

    return result

# File: refinement_pipeline.py
# Purpose: Perform high-fidelity photorealistic refinement of sketches using SDXL + ControlNet.

import os
import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

from dotenv import load_dotenv
load_dotenv()

# Constants for model paths on NVMe or local disk via env var
BASE_MODELS_DIR = os.getenv("SKAITCH_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
SDXL_PATH = os.path.join(BASE_MODELS_DIR, "sdxl")
CONTROLNET_PATH = os.path.join(BASE_MODELS_DIR, "controlnet-canny-sdxl")

# Global cache to avoid reloading on every click
# §4.1 fix: Threading lock prevents race conditions on concurrent Streamlit sessions
import threading
_refinement_pipe = None
_refinement_lock = threading.Lock()

def load_refinement_pipeline():
    """Lazy-load the SDXL ControlNet pipeline into VRAM."""
    global _refinement_pipe
    with _refinement_lock:
        if _refinement_pipe is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 1. Load ControlNet
            controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_PATH,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            # 2. Load SDXL with ControlNet
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                SDXL_PATH,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

            # Enable memory offloading instead of a full .to(device)
            if device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)  # Fallback for CPU-only environments

            _refinement_pipe = pipe

    return _refinement_pipe

def run_sdxl_refinement(
    sketch_pil: Image.Image,
    features: dict,
    extra_details: str = "",
    num_inference_steps: int = 40,
) -> Image.Image:
    """
    Takes an SDXL-generated sketch or a hand-drawn sketch and
    refines it into a photorealistic face using ControlNet.

    Enterprise fixes applied:
    - §3.3: enhance_sketch_for_edge_detection() is now called before edge detection.
    - §3.2: compute_edge_contrast() is fixed for fused (non-binary) edge maps.
    - §3.4: num_inference_steps is now a parameter (default raised to 40 for
             high-frequency forensic detail like pores, scars, fine hair).
    - §3.5: get_refinement_config() orchestrates all adaptive parameters.
    """
    import cv2
    from refinement_enhancements import (
        enhance_sketch_for_edge_detection,
        fused_edge_detection,
        get_refinement_config,
        RegionalGuidanceScaler,
    )

    # 1. §3.3 fix: Enhance sketch contrast BEFORE edge detection.
    # CLAHE + unsharp masking improves edge clarity for post-edit sketches
    # that have degraded contrast, directly mitigating §3.2 issues.
    enhanced_sketch = enhance_sketch_for_edge_detection(sketch_pil)

    # 2. Preprocess enhanced sketch with adaptive multi-modal edge detection.
    # Combines Canny + Sobel + Laplacian with contrast-adaptive thresholds.
    control_image_pil = fused_edge_detection(enhanced_sketch)
    control_image_np = np.array(control_image_pil)

    # 3. §3.5 fix: Use get_refinement_config() to compute ALL adaptive parameters
    # in one call — ControlNet scale, guidance, sharpening flags, steps.
    config = get_refinement_config(control_image_np)
    controlnet_scale = config["controlnet_conditioning_scale"]

    # 4. Build the Refinement prompt
    from prompt_builder import build_sdxl_refinement_prompt
    refinement_prompt, negative_prompt = build_sdxl_refinement_prompt(features, extra_details)

    # 5. Run Inference (§3.4 fix: steps now configurable, default 40)
    pipe = load_refinement_pipeline()

    result = pipe(
        prompt=refinement_prompt,
        image=control_image_pil,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=config["guidance_scale"],
    ).images[0]

    # 6. Apply region-specific post-processing sharpening to eyes and mouth.
    if config.get("apply_post_sharpening", True):
        result = RegionalGuidanceScaler.apply_regional_sharpening(result)

    return result

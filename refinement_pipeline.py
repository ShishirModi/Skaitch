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
_refinement_pipe = None

def load_refinement_pipeline():
    """Lazy-load the SDXL ControlNet pipeline into VRAM."""
    global _refinement_pipe
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
            pipe.to(device) # Fallback for CPU-only environments
            
        _refinement_pipe = pipe
        
    return _refinement_pipe

def run_sdxl_refinement(sketch_pil: Image.Image, features: dict, extra_details: str = "") -> Image.Image:
    """
    Takes an SDXL-generated sketch or a hand-drawn sketch and 
    refines it into a photorealistic face using ControlNet.
    Uses adaptive multi-modal edge detection, adaptive ControlNet conditioning scale,
    and region-specific post-generation sharpening from refinement_enhancements.py.
    """
    import cv2
    from refinement_enhancements import (
        fused_edge_detection,
        compute_adaptive_controlnet_scale,
        RegionalGuidanceScaler,
    )

    # 1. Preprocess sketch with adaptive multi-modal edge detection
    # Combines Canny + Sobel + Laplacian with contrast-adaptive thresholds.
    # Handles post-editing contrast degradation that would break fixed thresholds.
    control_image_pil = fused_edge_detection(sketch_pil)
    control_image_np = np.array(control_image_pil)  # grayscale numpy for scale computation

    # 2. Compute adaptive ControlNet conditioning scale based on actual edge quality
    # Sharp fresh sketches → ~0.75 (strict geometry), soft edited sketches → ~0.55 (more freedom)
    controlnet_scale = compute_adaptive_controlnet_scale(control_image_np)

    # 3. Build the Refinement prompt
    from prompt_builder import build_sdxl_refinement_prompt
    refinement_prompt, negative_prompt = build_sdxl_refinement_prompt(features, extra_details)

    # 4. Run Inference
    pipe = load_refinement_pipeline()

    result = pipe(
        prompt=refinement_prompt,
        image=control_image_pil,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=30,
        guidance_scale=9.0,
    ).images[0]

    # 5. Apply region-specific post-processing sharpening to eyes and mouth
    # Enhances perceptual quality of the two most forensically critical regions.
    result = RegionalGuidanceScaler.apply_regional_sharpening(result)

    return result

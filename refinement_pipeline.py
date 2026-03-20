# File: refinement_pipeline.py
# Purpose: Perform high-fidelity photorealistic refinement of sketches using SDXL + ControlNet.

import os
import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

# Constants for model paths on NVMe
SDXL_PATH = "/opt/dlami/nvme/models/sdxl"
CONTROLNET_PATH = "/opt/dlami/nvme/models/controlnet-canny-sdxl"

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
    """
    # 1. Preprocess the sketch for Canny
    # Convert to numpy for CV2 processing
    img_np = np.array(sketch_pil.convert("L"))
    
    # Run Canny edge detection
    # Generated sketches are usually clean, so we use relatively standard thresholds
    import cv2
    edges = cv2.Canny(img_np, 100, 200)
    
    # ControlNet-Canny expects white lines on black background
    # cv2.Canny already returns this format (edges are 255, background 0)
    control_image = Image.fromarray(edges)
    
    # 2. Build the "Refinement" prompt using the centralized prompt builder
    from prompt_builder import build_sdxl_refinement_prompt
    refinement_prompt, negative_prompt = build_sdxl_refinement_prompt(features, extra_details)

    # 3. Run Inference
    pipe = load_refinement_pipeline()
    
    # We use a relatively low guidance scale and moderate control strength
    # to allow the model to "fill in" the photorealistic details while
    # strictly following the sketch geometry.
    result = pipe(
        prompt=refinement_prompt,
        image=control_image,
        negative_prompt=negative_prompt,
        controlnet_conditioning_scale=0.65, # Lowered to allow prompt-driven eye color and textures
        num_inference_steps=30,
        guidance_scale=9.0,
    ).images[0]

    return result

# File: dfd_integration.py
# Purpose: Bridge the Skaitch Streamlit UI parameters to the PyTorch DeepFaceDrawing model (Xu-Justin implementation).

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Setup DeepFaceDrawing import path
dfd_dir = os.path.join(os.path.dirname(__file__), "external", "DeepFaceDrawing")
if dfd_dir not in sys.path:
    sys.path.append(dfd_dir)

# Global instance so we don't reload the model on every click
_dfd_model = None

def get_dfd_model():
    """Lazy load the PyTorch DeepFaceDrawing model."""
    global _dfd_model
    if _dfd_model is None:
        try:
            import models
        except ImportError:
            # If standard import fails, try relative import if possible or wait for download
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with modules used in Xu-Justin's inference script
        model = models.DeepFaceDrawing(
            CE=True, CE_encoder=True, CE_decoder=False,
            FM=True, FM_decoder=True,
            IS=True, IS_generator=True, IS_discriminator=False,
            manifold=False
        )
        
        # Weights are expected in external/DeepFaceDrawing/checkpoints/
        weights_path = os.path.join(dfd_dir, "checkpoints")
        if os.path.exists(weights_path):
            model.load(weights_path, map_location=device)
        
        model.to(device)
        model.eval()
        _dfd_model = model
        
    return _dfd_model

def run_dfd(image_pil: Image.Image, features: dict) -> Image.Image:
    """
    Inference pass for the PyTorch-based DeepFaceDrawing implementation.
    """
    model = get_dfd_model()
    if model is None:
        raise RuntimeError("DeepFaceDrawing model not initialized. Ensure external/DeepFaceDrawing is populated.")

    device = next(model.parameters()).device

    # 1. Preprocessing (matches Xu-Justin's transform_sketch)
    # Grayscale -> Resize 512 -> ToTensor
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    # 2. Inference
    with torch.no_grad():
        # result is a tensor (1, 3, 512, 512)
        result = model(input_tensor)
    
    # 3. Postprocessing (Tensor to PIL)
    # The models usually output in [0, 1] range
    res_img = result[0].cpu().clamp(0, 1)
    res_pil = transforms.ToPILImage()(res_img)
    
    return res_pil

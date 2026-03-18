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
    """Lazy load the PyTorch DeepFaceDrawing model with strict path priority."""
    global _dfd_model
    if _dfd_model is not None:
        return _dfd_model, None

    if not os.path.exists(dfd_dir):
        return None, f"Directory missing: {dfd_dir}. Run download_model.py manually."

    # Force sys.path priority for the external repo
    if dfd_dir not in sys.path:
        sys.path.insert(0, dfd_dir)
    
    try:
        # Check if 'models' is already loaded and if it's the wrong one
        if 'models' in sys.modules:
            m = sys.modules['models']
            if not hasattr(m, "DeepFaceDrawing"):
                del sys.modules['models']
        
        import models
        import importlib
        importlib.reload(models) 
        
        if not hasattr(models, "DeepFaceDrawing"):
            return None, "Module 'models' loaded but 'DeepFaceDrawing' class not found. Check if external/DeepFaceDrawing is correctly cloned."

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.DeepFaceDrawing(
            CE=True, CE_encoder=True, CE_decoder=False,
            FM=True, FM_decoder=True,
            IS=True, IS_generator=True, IS_discriminator=False,
            manifold=False
        )
        
        weights_path = os.path.join(dfd_dir, "checkpoints")
        if os.path.exists(weights_path):
            model.load(weights_path, map_location=device)
        else:
            return None, f"Weights folder missing: {weights_path}. Run download_model.py."
        
        model.to(device)
        model.eval()
        _dfd_model = model
        return _dfd_model, None
        
    except Exception as e:
        import traceback
        err_msg = f"Inference engine Error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {err_msg}")
        return None, err_msg

def run_dfd(image_pil: Image.Image, features: dict) -> Image.Image:
    """
    Inference pass for the PyTorch-based DeepFaceDrawing implementation.
    """
    model, error = get_dfd_model()
    if model is None:
        raise RuntimeError(f"DeepFaceDrawing failed to initialize: {error}")

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

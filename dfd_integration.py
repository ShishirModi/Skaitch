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
    if _dfd_model is None:
        if not os.path.exists(dfd_dir):
            return None

        # Force sys.path priority for the external repo
        if dfd_dir not in sys.path:
            sys.path.insert(0, dfd_dir)
        
        try:
            # Check if 'models' is already loaded and if it's the wrong one
            if 'models' in sys.modules:
                m = sys.modules['models']
                if not hasattr(m, "DeepFaceDrawing"):
                    # This is likely a system 'models' package (e.g. from torchvision or others)
                    # We need to temporarily remove it to import our specific one
                    del sys.modules['models']
            
            import models
            import importlib
            importlib.reload(models) # Ensure we get the one from dfd_dir
            
            if not hasattr(models, "DeepFaceDrawing"):
                print("❌ DeepFaceDrawing class still not found in 'models' module.")
                return None

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
            
            model.to(device)
            model.eval()
            _dfd_model = model
            
        except Exception as e:
            print(f"❌ Failed to load DeepFaceDrawing model: {e}")
            import traceback
            traceback.print_exc()
            return None
        
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

# File: dfd_integration.py
# Purpose: Bridge the Skaitch Streamlit UI parameters to the DeepFaceDrawing Jittor model.

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Setup DeepFaceDrawing import path
dfd_dir = os.path.join(os.path.dirname(__file__), "external", "DeepFaceDrawing")
sys.path.append(dfd_dir)

# Global instance so we don't reload the Jittor model on every click
_combine_model = None

def get_combine_model():
    """Lazy load the CombineModel to save memory until clicked."""
    global _combine_model
    if _combine_model is None:
        import jittor as jt
        # Attempt to enable CUDA if available, but default to CPU since Streamlit is likely CPU bound
        try:
            if jt.has_cuda:
                jt.flags.use_cuda = 1
            else:
                jt.flags.use_cuda = 0
        except Exception:
            jt.flags.use_cuda = 0
        
        from CombineModel_jt import CombineModel

        # Before instantiating the model, we MUST switch the CWD temporarily because
        # DeepFaceDrawing loads checkpoint paths relative to its own root directory
        original_cwd = os.getcwd()
        os.chdir(dfd_dir)
        try:
            _combine_model = CombineModel()
        finally:
            os.chdir(original_cwd)
    return _combine_model

def calculate_weights(features: dict) -> list[float]:
    """
    Map Streamlit selected features to DeepFaceDrawing weights.
    Returns: [eye1, eye2, nose, mouth, (bg/face_base)]
    """
    # Default weights [eye1, eye2, nose, mouth, base]
    weights = [0.8, 0.8, 0.8, 0.8, 0.8]

    # Adjust based on eyes
    eyes = features.get("Eyes", "")
    if eyes in ["Wide-set", "Close-set"]:
        weights[0] += 0.15 # Heavier emphasis on eye placement
        weights[1] += 0.15
        
    # Adjust based on nose
    nose = features.get("Nose", "")
    if nose in ["Broad", "Wide bridge", "Bulbous"]:
        weights[2] += 0.20 # Force wider nose rendering
    elif nose in ["Narrow", "Aquiline"]:
        weights[2] -= 0.10

    # Adjust based on mouth
    mouth = features.get("Mouth / Lips", "")
    if mouth in ["Full", "Heavy lower lip", "Heavy upper lip"]:
        weights[3] += 0.15
    elif mouth == "Thin":
        weights[3] -= 0.10

    # Ensure weights are between 0 and 1
    return [max(0.1, min(w, 1.0)) for w in weights]

def run_dfd(image_pil: Image.Image, features: dict) -> Image.Image:
    """
    Takes a Stable Diffusion generated sketch (PIL Image) and 
    processes it through DeepFaceDrawing to produce a photorealistic face.
    """
    # 1. Convert PIL to OpenCV format (RGB -> BGR)
    mat_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # 2. Resize to 512x512 as expected by DeepFaceDrawing
    mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    
    # 3. Load model
    model = get_combine_model()

    # 4. Map Gender (DeepFaceDrawing expects 1 for Male, 0 for Female)
    gender = features.get("Gender", "Male")
    model.sex = 1 if gender == "Male" else 0

    # 5. Apply calculated weights
    eye1, eye2, nose, mouth, base = calculate_weights(features)
    model.part_weight['eye1'] = eye1
    model.part_weight['eye2'] = eye2
    model.part_weight['nose'] = nose
    model.part_weight['mouth'] = mouth
    model.part_weight[''] = base
    
    # 6. Run Inference
    # predict_shadow() generates the image and stores it in model.generated
    original_cwd = os.getcwd()
    os.chdir(dfd_dir) # Required for the inner logic of predict_shadow which assumes local scope
    try:
        model.predict_shadow(mat_img)
    finally:
        os.chdir(original_cwd)
    
    # 7. Convert output BGR back to PIL RGB
    output_rgb = cv2.cvtColor(model.generated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)

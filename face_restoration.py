# File: face_restoration.py
# Purpose: High-fidelity face restoration using CodeFormer.

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Setup CodeFormer import path
codeformer_dir = os.path.join(os.path.dirname(__file__), "external", "CodeFormer")
if codeformer_dir not in sys.path:
    sys.path.insert(0, codeformer_dir)

# Global caches (stored on CPU to save VRAM)
_face_helper = None
_codeformer_net = None

def load_codeformer_models():
    """Lazy load CodeFormer and FaceRestoreHelper to CPU."""
    global _face_helper, _codeformer_net
    if _codeformer_net is not None:
        return _face_helper, _codeformer_net

    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    
    # Surgical import from the cloned repository's local basicsr
    if codeformer_dir not in sys.path:
        sys.path.insert(0, codeformer_dir)
        
    from basicsr.archs.codeformer_arch import CodeFormer

    device = torch.device("cpu")
    
    # 1. Initialize FaceRestoreHelper (for detection/alignment)
    _face_helper = FaceRestoreHelper(
        upscale_factor=1, 
        face_size=512, 
        crop_ratio=(1, 1), 
        det_model='retinaface_resnet50', 
        save_ext='png', 
        device=device
    )

    # 2. Initialize CodeFormer Network
    _codeformer_net = CodeFormer(
        dim_embd=512, 
        codebook_size=1024, 
        n_head=8, 
        n_layers=9, 
        connect_list=['32', '64', '128', '256']
    ).to(device)
    
    # 3. Load weights
    base_models_dir = os.environ.get("SKAITCH_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
    ckpt_path = os.path.join(base_models_dir, "codeformer", "codeformer.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(codeformer_dir, "weights", "CodeFormer", "codeformer.pth")
        
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
        _codeformer_net.load_state_dict(checkpoint)
        _codeformer_net.eval()
    else:
        raise FileNotFoundError(f"CodeFormer weights not found at {ckpt_path}")

    return _face_helper, _codeformer_net

def run_codeformer(img_pil: Image.Image, fidelity: float = 0.5) -> Image.Image:
    """
    Apply CodeFormer restoration to a PIL image.
    fidelity: 0.0 (max restoration) to 1.0 (max fidelity to original).
    """
    helper, net = load_codeformer_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Move models to GPU explicitly for inference
        if device.type == "cuda":
            net = net.to(device)
            helper.face_det = helper.face_det.to(device)
            if hasattr(helper, 'face_parse') and helper.face_parse is not None:
                helper.face_parse = helper.face_parse.to(device)
        
        from basicsr.utils import img2tensor, tensor2img
        
        # Convert PIL to CV2 (BGR)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Clean previous faces
        helper.clean_all()
        helper.read_image(img_cv)
        
        # Detect and align faces
        helper.get_face_endpoints(5) # max faces
        helper.align_warp_face()
        
        # Restore each face
        for cropped_face in helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            face_t = face_t.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = net(face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(0, 1))
            
            restored_face = restored_face.astype('uint8')
            helper.add_restored_face(restored_face)
        
        # Paste faces back into image
        helper.get_inverse_affine(None)
        restored_img = helper.paste_faces_to_input_image()
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        
    finally:
        # Prevent VRAM Leak: Move models back to CPU 
        if device.type == "cuda":
            net = net.to("cpu")
            helper.face_det = helper.face_det.to("cpu")
            if hasattr(helper, 'face_parse') and helper.face_parse is not None:
                helper.face_parse = helper.face_parse.to("cpu")
            torch.cuda.empty_cache()

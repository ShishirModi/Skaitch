import os
import subprocess
import urllib.request
from dotenv import load_dotenv

load_dotenv()

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

# New PyTorch DFD Repository and Weights
DFD_REPO_URL = "https://github.com/Xu-Justin/DeepFaceDrawing.git"
DFD_WEIGHTS_REPO_URL = "https://github.com/Xu-Justin/DeepFaceDrawing-Weight.git"

SDXL_OUTPUT_DIR = "/opt/dlami/nvme/models/sdxl"
CODEFORMER_OUTPUT_DIR = "/opt/dlami/nvme/models/codeformer"
DFD_DIR = os.path.join(os.path.dirname(__file__), "external", "DeepFaceDrawing")

def is_model_downloaded(directory):
    if not os.path.exists(directory):
        return False
    # Check if directory contains at least a few files (like model weights)
    files = os.listdir(directory)
    return len(files) > 2

def check_and_download_dfd():
    """Clones the PyTorch DFD repo and its weights from GitHub."""
    os.makedirs(os.path.dirname(DFD_DIR), exist_ok=True)
    
    # 1. Clone the Source Code Repository if missing
    if not os.path.exists(DFD_DIR):
        print(f"Cloning DeepFaceDrawing (PyTorch) source from {DFD_REPO_URL}...")
        subprocess.run(["git", "clone", DFD_REPO_URL, DFD_DIR], check=True)
    else:
        print(f"✅ DeepFaceDrawing source already exists at {DFD_DIR}.")

    # 2. Clone/Download Weights
    # The PyTorch version expects weights in a 'checkpoints' directory
    weights_dir = os.path.join(DFD_DIR, "checkpoints")
    if not os.path.exists(weights_dir) or len(os.listdir(weights_dir)) < 3:
        print(f"Cloning DeepFaceDrawing Weights from {DFD_WEIGHTS_REPO_URL}...")
        # Clone into a temporary dir and move contents to avoid nested repo issues
        tmp_weights = os.path.join(os.path.dirname(DFD_DIR), "dfd_weights_tmp")
        if os.path.exists(tmp_weights):
            subprocess.run(["rm", "-rf", tmp_weights])
        subprocess.run(["git", "clone", DFD_WEIGHTS_REPO_URL, tmp_weights], check=True)
        
        os.makedirs(weights_dir, exist_ok=True)
        # Move all contents from tmp_weights/ to weights_dir/
        for item in os.listdir(tmp_weights):
            if item == ".git": continue
            s = os.path.join(tmp_weights, item)
            d = os.path.join(weights_dir, item)
            subprocess.run(["mv", s, d])
        
        subprocess.run(["rm", "-rf", tmp_weights])
        print(f"✅ DeepFaceDrawing weights installed successfully in {weights_dir}.")
    else:
        print(f"✅ DeepFaceDrawing weights already exist in {weights_dir}.")

from huggingface_hub import snapshot_download

def check_and_download_models():
    """Checks if models are present on the NVMe drive and downloads them if not."""
    os.makedirs(SDXL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CODEFORMER_OUTPUT_DIR, exist_ok=True)
    
    if not is_model_downloaded(SDXL_OUTPUT_DIR):
        print(f"Downloading {SDXL_MODEL_ID} to {SDXL_OUTPUT_DIR}...")
        snapshot_download(repo_id=SDXL_MODEL_ID, local_dir=SDXL_OUTPUT_DIR)
    else:
        print(f"✅ SDXL model already exists at {SDXL_OUTPUT_DIR}.")
        
    # Check CodeFormer (GitHub Release)
    codeformer_path = os.path.join(CODEFORMER_OUTPUT_DIR, "codeformer.pth")
    if not os.path.exists(codeformer_path):
        print(f"Downloading CodeFormer to {codeformer_path}...")
        urllib.request.urlretrieve(CODEFORMER_MODEL_URL, codeformer_path)
    else:
        print(f"✅ CodeFormer model already exists at {CODEFORMER_OUTPUT_DIR}.")

    # Check DeepFaceDrawing phase 2 models
    check_and_download_dfd()

if __name__ == "__main__":
    # When run directly, just execute the checks and downloads
    check_and_download_models()

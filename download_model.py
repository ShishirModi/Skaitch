import os
import subprocess
import urllib.request
from dotenv import load_dotenv

load_dotenv()

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
CODEFORMER_REPO_URL = "https://github.com/sczhou/CodeFormer.git"

BASE_MODELS_DIR = os.getenv("SKAITCH_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

SDXL_OUTPUT_DIR = os.path.join(BASE_MODELS_DIR, "sdxl")
CODEFORMER_OUTPUT_DIR = os.path.join(BASE_MODELS_DIR, "codeformer")
CODEFORMER_DIR = os.path.join(os.path.dirname(__file__), "external", "CodeFormer")
CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_OUTPUT_DIR = os.path.join(BASE_MODELS_DIR, "controlnet-canny-sdxl")

def is_model_downloaded(directory):
    if not os.path.exists(directory):
        return False
    # Check if directory contains at least a few files
    files = os.listdir(directory)
    return len(files) > 2

def check_and_download_controlnet():
    """Downloads the SDXL Canny ControlNet weights to the NVMe drive."""
    os.makedirs(CONTROLNET_OUTPUT_DIR, exist_ok=True)
    
    if not is_model_downloaded(CONTROLNET_OUTPUT_DIR):
        print(f"Downloading {CONTROLNET_MODEL_ID} to {CONTROLNET_OUTPUT_DIR}...")
        snapshot_download(repo_id=CONTROLNET_MODEL_ID, local_dir=CONTROLNET_OUTPUT_DIR)
        print(f"✅ ControlNet model installed successfully in {CONTROLNET_OUTPUT_DIR}.")
    else:
        print(f"✅ ControlNet model already exists at {CONTROLNET_OUTPUT_DIR}.")

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
        print(f"Downloading CodeFormer weights to {codeformer_path}...")
        urllib.request.urlretrieve(CODEFORMER_MODEL_URL, codeformer_path)
    else:
        print(f"✅ CodeFormer weights already exist.")

    # Check CodeFormer Repository
    if not os.path.exists(CODEFORMER_DIR):
        print(f"Cloning CodeFormer repository from {CODEFORMER_REPO_URL}...")
        os.makedirs(os.path.dirname(CODEFORMER_DIR), exist_ok=True)
        subprocess.run(["git", "clone", CODEFORMER_REPO_URL, CODEFORMER_DIR], check=True)
        
        # Move weights to the repo's expected location
        repo_weights_dir = os.path.join(CODEFORMER_DIR, "weights", "CodeFormer")
        os.makedirs(repo_weights_dir, exist_ok=True)
        import shutil
        shutil.copy(codeformer_path, os.path.join(repo_weights_dir, "codeformer.pth"))
        print(f"✅ CodeFormer repository and weights installed successfully.")
    else:
        print(f"✅ CodeFormer repository already exists.")

    # Check ControlNet refinement models
    check_and_download_controlnet()

if __name__ == "__main__":
    # When run directly, just execute the checks and downloads
    check_and_download_models()

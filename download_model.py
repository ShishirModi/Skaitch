import os
import subprocess
import urllib.request
from dotenv import load_dotenv

load_dotenv()

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CODEFORMER_MODEL_URL = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

SDXL_OUTPUT_DIR = "/opt/dlami/nvme/models/sdxl"
CODEFORMER_OUTPUT_DIR = "/opt/dlami/nvme/models/codeformer"
DFD_PARAMS_DIR = os.path.join(os.path.dirname(__file__), "external", "DeepFaceDrawing", "Params")

def is_model_downloaded(directory):
    if not os.path.exists(directory):
        return False
    # Check if directory contains at least a few files (like model weights)
    files = os.listdir(directory)
    return len(files) > 2

def check_and_download_dfd():
    """Checks if DeepFaceDrawing weights exist, optionally downloads them, or alerts user."""
    os.makedirs(DFD_PARAMS_DIR, exist_ok=True)
    
    # Check if any .pkl files exist in the Params directory
    has_pkl = any(f.endswith('.pkl') for f in os.listdir(DFD_PARAMS_DIR))
    if has_pkl:
        print(f"✅ DeepFaceDrawing weights found in {DFD_PARAMS_DIR}.")
        return

    # If missing, try to download via ENV direct URL if provided
    direct_url = os.environ.get("DFD_WEIGHTS_DIRECT_URL")
    if direct_url and direct_url.strip():
        print(f"Downloading DeepFaceDrawing weights from {direct_url}...")
        try:
            # Simple download assuming ZIP or single file. If it's a ZIP, it would need extraction.
            filename = direct_url.split("/")[-1] or "dfd_weights.zip"
            dest = os.path.join(DFD_PARAMS_DIR, filename)
            urllib.request.urlretrieve(direct_url, dest)
            print(f"✅ Downloaded to {dest}. Note: You may need to unzip this manually.")
            return
        except Exception as e:
            print(f"⚠️ Failed to download from direct URL: {e}")

    # Fallback alert instruction
    print("\n" + "="*80)
    print("🚨 ACTION REQUIRED: MISSING DEEPFACEDRAWING WEIGHTS 🚨")
    print("The official DeepFaceDrawing pre-trained models are hosted on Baidu Pan,")
    print("which cannot be downloaded programmatically without an account session.")
    print("Please download them manually:")
    print("Link: https://pan.baidu.com/s/1f1S9t4T5X5J0CDZ7AqTfMg")
    print("Code: wiu9")
    print(f"Extract the .pkl files into: {DFD_PARAMS_DIR}")
    print("Or provide a direct URL mirroring the weights in the .env file:")
    print("DFD_WEIGHTS_DIRECT_URL='https://you-mirror-link/weights.zip'")
    print("="*80 + "\n")

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

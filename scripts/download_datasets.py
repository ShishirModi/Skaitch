import os
import argparse
import subprocess

def download_cufs(output_dir):
    cufs_dir = os.path.join(output_dir, "CUFS")
    os.makedirs(cufs_dir, exist_ok=True)
    try:
        # Instead of shelling out to the kaggle CLI which often has PATH issues, use the native python kagglehub library
        import kagglehub
        import shutil
        
        print("[INFO] Attempting to download CUFS dataset via kagglehub...")
        path = kagglehub.dataset_download("arbazhussain/cuhk-face-sketch-database-cufs")
        print(f"[SUCCESS] Dataset cached at: {path}")
        
        # Copy from kagglehub cache to our defined cufs_dir
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(cufs_dir, item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)
        print(f"[SUCCESS] CUFS extracted and moved to {cufs_dir}")
        
    except ImportError:
        print("[WARNING] 'kagglehub' module not found. Falling back to Kaggle CLI...")
        try:
            subprocess.run(["kaggle", "datasets", "download", "-d", "arbazhussain/cuhk-face-sketch-database-cufs", "-p", cufs_dir, "--unzip"], check=True)
            print(f"[SUCCESS] CUFS downloaded and extracted to {cufs_dir}")
        except Exception as e:
            print(f"[ERROR] Automated download for CUFS failed: {e}")
            print("Please manually download from Kaggle (arbazhussain/cuhk-face-sketch-database-cufs) or MMLAB.")
            print(f"Extract contents to: {cufs_dir}")
            
    except Exception as e:
        print(f"[WARNING] Automated download for CUFS failed: {e}")
        print("Please manually download from Kaggle (arbazhussain/cuhk-face-sketch-database-cufs) or MMLAB.")
        print(f"Extract contents to: {cufs_dir}")

def download_iiitd(output_dir):
    iiitd_dir = os.path.join(output_dir, "IIIT-D_Sketch")
    os.makedirs(iiitd_dir, exist_ok=True)
    print("\n[INFO] Downloading IIIT-D Sketch dataset...")
    # This usually requires manual request from http://iab-rubric.org/resources/sketch-database
    print(">> IIIT-D Sketch Database typically requires an explicit request via email.")
    print(">> Please visit: http://iab-rubric.org/resources/sketch-database")
    print(f">> Extract the obtained dataset to: {iiitd_dir}")

def download_prip_vsgc(output_dir):
    prip_dir = os.path.join(output_dir, "PRIP-VSGC")
    os.makedirs(prip_dir, exist_ok=True)
    print("\n[INFO] Downloading PRIP-VSGC dataset...")
    print(">> PRIP-VSGC datasets are provided for academic research and may require manual access.")
    print(f">> Please place the extracted dataset folders into: {prip_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download Forensic Composite Datasets (CUFS, IIIT-D, PRIP-VSGC)")
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw'), help='Directory to save raw datasets')
    args = parser.parse_args()
    
    print(f"Target directory for raw data: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    download_cufs(args.output_dir)
    download_iiitd(args.output_dir)
    download_prip_vsgc(args.output_dir)
    
    print("\n[INFO] Dataset provisioning setup completed.")
    print("Please follow any manual instructions printed above before running `scripts/preprocess.py`.")

if __name__ == "__main__":
    main()

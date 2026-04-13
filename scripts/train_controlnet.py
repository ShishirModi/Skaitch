import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Diffusers SDXL ControlNet Training")
    parser.add_argument('--dataset_dir', type=str, default='../data/cleaned_dataset/controlnet_512')
    parser.add_argument('--output_dir', type=str, default='../models/controlnet_sketch_tmp')
    parser.add_argument('--diffusers_dir', type=str, default='./diffusers', help='Path to diffusers repository')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    
    script_path = os.path.join(args.diffusers_dir, "examples", "controlnet", "train_controlnet_sdxl.py")
    if not os.path.exists(script_path):
        print(f"[WARNING] Diffusers example script not found at {script_path}")
        print(">> Please clone diffusers: git clone https://github.com/huggingface/diffusers.git")

    command = [
        "accelerate", "launch",
        script_path if os.path.exists(script_path) else "train_controlnet_sdxl.py",
        f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        f"--output_dir={args.output_dir}",
        f"--dataset_name={args.dataset_dir}",
        "--resolution=512",
        "--learning_rate=1e-5",
        "--train_batch_size=2",
        "--gradient_accumulation_steps=2",
        "--max_train_steps=30000",
        "--use_8bit_adam",
        "--mixed_precision=fp16",
        "--proportion_empty_prompts=0.2",
        "--set_grads_to_none",
        "--controlnet_model_name_or_path=skaitch_sketch"
    ]
    
    print("\n[INFO] Target: Sketch-conditioned ControlNet from scratch for SDXL")
    print("[INFO] Target Parameters: 512px, BS=2, Grad=2, LR=1e-5, 30k steps, fp16")
    print(f"[INFO] Time Estimate on Google Colab Pro / A100: ~24 - 36 hours")
    print("\nDiffusers command to execute:\n" + " ".join(command) + "\n")
    print(">> NOTE: Make sure you have a `metadata.jsonl` in your dataset directory mapping:")
    print("   {'text': prompt, 'image': 'photo.png', 'conditioning_image': 'sketch.png'}")
    print(">> When training finishes, move and rename the safetensors model to `../models/controlnet_sketch.safetensors`\n")
    
    print("If you are inside Colab/A100, run the printed command via !accelerate launch ...\n")
    print("Do you want to attempt running it locally now? (y/n)")
    try:
        ans = input().lower()
        if ans == 'y':
            subprocess.run(command, check=True)
    except FileNotFoundError:
        print("[ERROR] Accelerate not found. Please install the diffusers environment.")
    except Exception as e:
        print(f"Execution handling: {e}")

if __name__ == "__main__":
    main()

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Kohya_SS SDXL LoRA Training")
    parser.add_argument('--dataset_dir', type=str, default='../data/cleaned_dataset/train')
    parser.add_argument('--output_dir', type=str, default='../models')
    parser.add_argument('--kohya_dir', type=str, default='./kohya_ss', help='Path to kohya_ss repository')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    script_path = os.path.join(args.kohya_dir, "sdxl_train_network.py")
    if not os.path.exists(script_path):
        print(f"[WARNING] Kohya_ss script not found at {script_path}")
        print(">> Please clone Kohya_ss: git clone https://github.com/bmaltais/kohya_ss.git")
        print(">> Or provide the correct path via --kohya_dir")

    command = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=2",
        script_path if os.path.exists(script_path) else "sdxl_train_network.py",
        f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        f"--train_data_dir={args.dataset_dir}",
        f"--output_dir={args.output_dir}",
        "--output_name=skaitch_lora",
        "--save_model_as=safetensors",
        "--resolution=768,768",
        "--train_batch_size=2",
        "--learning_rate=1e-4",
        "--network_module=networks.lora",
        "--network_dim=64",
        "--network_alpha=32",
        "--optimizer_type=AdamW",
        "--lr_scheduler=cosine",
        "--max_train_steps=6000",
        "--mixed_precision=fp16",
        "--xformers",
        "--cache_latents",
        "--cache_text_encoder_outputs"
    ]
    
    print("\n[INFO] Target: SDXL Base + Custom Sketch Domain LoRA")
    print("[INFO] Target Parameters: 768px, BS=2, LR=1e-4, 6000 steps, fp16")
    print(f"[INFO] Time Estimate on Google Colab T4: ~3 hours")
    print("\nKohya command to execute:\n" + " ".join(command) + "\n")
    
    print("If you are inside Colab, run the printed command via !accelerate launch ...\n")
    print("Do you want to attempt running it locally now? (y/n)")
    try:
        ans = input().lower()
        if ans == 'y':
            subprocess.run(command, check=True)
    except FileNotFoundError:
        print("[ERROR] Accelerate not found. Please install the kohya environment.")
    except Exception as e:
        print(f"Execution handling: {e}")

if __name__ == "__main__":
    main()

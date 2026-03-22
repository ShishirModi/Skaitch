import os
import torch
from diffusers import StableDiffusionXLPipeline
from prompt_builder import build_sdxl_forensic_prompt, FORENSIC_DEFAULTS

def main():
    print("Testing Forensic V2.1 Generation...")
    
    # 1. Grab defaults and generate prompt
    num_inference_steps = FORENSIC_DEFAULTS["num_inference_steps"]
    guidance_scale = FORENSIC_DEFAULTS["guidance_scale"]
    
    sample_features = {
        "Gender": "Male",
        "Age range": "26–35",
        "Face shape": "Square",
        "Eyes": "Deep-set",
        "Eyebrows": "Bushy",
        "Nose": "Broad",
        "Mouth / Lips": "Thin",
        "Jawline": "Strong",
        "Hair style": "Short cropped",
        "Hair color": "Dark brown",
        "Facial hair": "Stubble",
        "Ethnicity": "Hispanic / Latino",
        "Skin tone": "Olive",
        "Eye color": "Brown",
        "Spectacles": "None",
        "Spectacles Tint": "None",
        "Distinguishing marks": "Scar on left cheek",
    }
    
    prompt, negative_prompt = build_sdxl_forensic_prompt(
        sample_features,
        style="Pencil sketch",
        extra_details="",
    )
    
    print("\n[PROMPT]")
    print(prompt)
    print("\n[NEGATIVE PROMPT]")
    print(negative_prompt)
    
    # 2. Check model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.environ.get("SKAITCH_MODEL_DIR", os.path.join(base_dir, "models", "sdxl"))
    
    if not os.path.exists(model_path):
        print(f"\nModel not found at: {model_path}")
        print("Please run 'python download_model.py' first.")
        return

    # 3. Load Pipeline
    print("\nLoading SDXL Pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # 4. Generate
    print("\nGenerating...")
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    output_path = os.path.join(base_dir, "test_forensic.png")
    image.save(output_path)
    print(f"\nSaved test image to {output_path}")

if __name__ == "__main__":
    main()
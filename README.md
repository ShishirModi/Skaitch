# Skaitch
**Forensic Composite Suite · Generative Morphology Matching Pipeline**

Skaitch is a professional-grade forensic sketching and photorealistic refinement pipeline. It utilizes Stable Diffusion XL (SDXL) and ControlNet to translate categorical facial feature selections into high-fidelity investigative composites.

## Core Features (V2.3 Architecture)

*   **Phase I (Narrative Prompting):** Leverages a sophisticated prompt builder to construct grammatical narratives from categorical facial features. The SDXL Base 1.0 model parses this narrative to generate structural sketch variants in 4 styles: **Pencil sketch**, **Charcoal sketch**, **Police composite**, and **Forensic artist rendering**. Enhanced with ethnicity-specific anatomical grounding, adaptive guidance scaling based on feature complexity, and a CLIP token budget guard to prevent silent prompt truncation.
*   **Batched Generation:** All three sketch variants are generated in a single batched forward pass (`num_images_per_prompt=3`) with diversified seeds using large prime offsets, reducing wall-clock time by ~2× while producing meaningfully diverse compositions.
*   **Phase II (ControlNet Refinement):** Uses the `diffusers/controlnet-canny-sdxl-1.0` pipeline to elevate the finalized sketch into a photorealistic composite. Sketch contrast is pre-enhanced via CLAHE before multi-modal edge fusion (Canny + Sobel + Laplacian). The adaptive ControlNet conditioning scale, configurable inference steps (default 40), and region-specific post-processing sharpening are now fully wired into the live pipeline path.
*   **Domain-Specialized Generative Pipeline:** A unified automated execution environment that downloads forensic databases (CUFS, IIIT-D), normalizes faces to structural constraints, and dynamically tunes **custom LoRA (skaitch_lora)** identifying stylistic mappings and a **Sketch ControlNet (controlnet_sketch)** ensuring rigid unyielding geometric faithfulness—removing dependencies on canonical Canny maps.
*   **Fully Automated Setup:** Running the application automatically initializes rigorous dependency verification, optionally triggering dataset synthesis (`preprocess.py`) and LoRA/ControlNet acceleration jobs (`train_lora.py`, `train_controlnet.py`) if the weights do not exist locally.
*   **Targeted Mask Editing (Img2Img + Compositing):** Uses `StableDiffusionXLImg2ImgPipeline` with post-generation mask compositing, ensuring the base UNet's 4-channel architecture is correctly matched. The full `prepare_enhanced_inpaint_inputs()` pipeline (feathering, auto-dilation, adaptive strength, graduated strength maps mapping matrix) is wired into the UI loop preventing pixel drift in untouched sections.
*   **Automated Face Restoration (Phase II only):** CodeFormer is reserved exclusively for the photorealistic Phase II output, where its VQGAN codebook operates in the correct photorealistic domain. A face-detection gate prevents silent failures when no faces are detected.
*   **Enterprise Stability:** Threading locks protect global pipeline caches against race conditions in multi-session deployments. The base pipeline is fully unloaded before Phase II to prevent VRAM deadlocks. Edit history is capped at 15 entries to prevent unbounded memory growth.

## System Requirements

Skaitch is tailored for operation on standard cloud GPU instances with constrained VRAM. Memory offloading ensures stability.

*   **GPU:** Minimum 16GB VRAM (NVIDIA T4, RTX 3080, or equivalent).
*   **OS:** Linux (Ubuntu 22.04 LTS recommended), macOS Apple Silicon (M-series), or Windows (via WSL2).
*   **RAM:** 32GB System RAM.
*   **Storage:** 20GB+ fast NVMe SSD space for model weights.

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
```

**2. Configure Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Copy `.env.example` to `.env` and set your preferred paths:
```bash
cp .env.example .env
```
Ensure `SKAITCH_MODEL_DIR` points to an SSD path to ensure rapid model swapping. Set `ADMIN_PASSWORD` to secure the settings page and gallery.

**4. Download Model Weights & Execute Training Pipelines**
Skaitch relies on local models to ensure privacy. Run the automated downloader:
```bash
python download_model.py
```
*This downloads SDXL Base and CodeFormer to your configured directory.*

On initial startup, `app.py` will dynamically check for the specialized forensic pipelines (`skaitch_lora.safetensors` and `controlnet_sketch.safetensors`). If they are missing, it will automatically run data generation across CUFS arrays and initialize Diffusers training processes sequentially.

> [!NOTE]
> **Dataset Fetching:** The system automatically downloads the CUFS dataset using `kagglehub` ([arbazkhan971/cuhk-face-sketch-database-cufs](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs)). However, the **[IIIT-D Sketch](https://iab-rubric.org/index.php/iiit-d-sketch-database)** database resides behind an academic request wall. Operators must manually extract it into `data/raw/` prior to running the pipeline.

## Execution

Launch the primary Streamlit interface:
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`.

## UI/UX Design System
Skaitch uses a tailored CSS overlay enforcing a clean, high-contrast B2B SaaS aesthetic. 
- **Light Mode Enforced:** Independent of browser or OS settings for uncompromising accuracy when evaluating visual aids.
- **Color Palette:** Charcoal (`#1C1C1E`) for text, Amber (`#F59E0B`) for action items, and Ghost White (`#FAFAFA`) for elevated card surfaces.

## Repository Structure

*   `app.py`: Streamlit orchestration, auto-launching background training routines (LoRA, ControlNet), session state management, and thread-safe pipeline lifecycle.
*   `prompt_builder.py`: Maps UI dropdowns to narrative strings avoiding random comma delimiters. Strict enforcement of narrative-grammar structure over canonical AI token tags, enforcing specific CLIP constraints natively.
*   `download_model.py`: Automates environment validation and canonical model caching.
*   `face_restoration.py`: Wraps CodeFormer inference with cached device detection.
*   `refinement_pipeline.py`: Stage II Phase rendering loading the local custom `controlnet_sketch` model to enforce structure.
*   `sketch_refiner.py`: Implements SDXL Img2Img editing mapped cleanly onto `apply_graduated_strength_to_image` to support attenuated feathering on regional masks.
*   `visual_aids.py`: Inlines raw SVGs providing anatomical reference for operators.
*   `.streamlit/config.toml`: Enforces light mode and disables CORS/XSRF to support Cloudflare Tunnel deployments.
*   `inpaint_enhancements.py`: Fully integrated enhancement module providing mask feathering, graduated strength maps, prompt-based region inference, adaptive inpaint strength auto-tuning, and improved difference blending with soft attenuation curve for stable cascading edits.
*   `refinement_enhancements.py`: Fully integrated enhancement module providing adaptive ControlNet conditioning scale (fixed for non-binary fused edge maps), multi-modal edge fusion (Canny + Sobel + Laplacian), adaptive Canny threshold selection, CLAHE sketch pre-processing, and region-specific post-generation sharpening for eyes and mouth.

## License
Skaitch is licensed under the MIT License. See `LICENSE` for more information.

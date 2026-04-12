# Skaitch
**Forensic Composite Suite · Generative Morphology Matching Pipeline**

Skaitch is a professional-grade forensic sketching and photorealistic refinement pipeline. It utilizes Stable Diffusion XL (SDXL) and ControlNet to translate categorical facial feature selections into high-fidelity investigative composites.

## Core Features (V2.3 Architecture)

*   **Phase I (Narrative Prompting):** Leverages a sophisticated prompt builder to construct grammatical narratives from categorical facial features. The SDXL Base 1.0 model parses this narrative to generate structural sketch variants in 4 styles: **Pencil sketch**, **Charcoal sketch**, **Police composite**, and **Forensic artist rendering**. Enhanced with ethnicity-specific anatomical grounding, adaptive guidance scaling based on feature complexity, and a CLIP token budget guard to prevent silent prompt truncation.
*   **Batched Generation:** All three sketch variants are generated in a single batched forward pass (`num_images_per_prompt=3`) with diversified seeds using large prime offsets, reducing wall-clock time by ~2× while producing meaningfully diverse compositions.
*   **Phase II (ControlNet Refinement):** Uses the `diffusers/controlnet-canny-sdxl-1.0` pipeline to elevate the finalized sketch into a photorealistic composite. Sketch contrast is pre-enhanced via CLAHE before multi-modal edge fusion (Canny + Sobel + Laplacian). The adaptive ControlNet conditioning scale, configurable inference steps (default 40), and region-specific post-processing sharpening are now fully wired into the live pipeline path.
*   **Targeted Mask Editing (Img2Img + Compositing):** Uses `StableDiffusionXLImg2ImgPipeline` with post-generation mask compositing, ensuring the base UNet's 4-channel architecture is correctly matched. The full `prepare_enhanced_inpaint_inputs()` pipeline (feathering, auto-dilation, adaptive strength, graduated strength maps) is wired in. Aspect-ratio-aware canvas dimensions prevent coordinate skew on non-square resolutions.
*   **Automated Face Restoration (Phase II only):** CodeFormer is now reserved exclusively for the photorealistic Phase II output, where its VQGAN codebook operates in the correct domain. A face-detection gate prevents silent failures when no faces are detected.
*   **Enterprise Stability:** Threading locks protect global pipeline caches against race conditions in multi-session deployments. The base pipeline is fully unloaded before Phase II to prevent VRAM deadlocks. Edit history is capped at 15 entries to prevent unbounded memory growth. Model download integrity is validated by total file size.

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

**4. Download Model Weights**
Skaitch relies on local models to ensure privacy. Run the automated downloader:
```bash
python download_model.py
```
*This downloads SDXL Base, the SDXL Canny ControlNet, and CodeFormer to your configured directory.*

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

*   `app.py`: Streamlit orchestration, session state management, UI logic, and thread-safe pipeline lifecycle (load/unload with `threading.Lock`). Implements batched generation, adaptive guidance, edit history capping, and VRAM-safe Phase I→II transitions.
*   `prompt_builder.py`: Maps UI dropdowns to narrative strings and applies SDXL syntax boosters. Includes `build_sdxl_forensic_prompt()` (with CLIP token budget guard), `build_sdxl_refinement_prompt()`, `build_edit_prompt()`, `compute_adaptive_guidance_scale()` (now wired into the live pipeline), ethnicity-specific anatomical grounding, and prompt-based mask region inference with word-boundary-aware matching (list-of-tuples structure, no duplicate-key collisions).
*   `download_model.py`: Automates environment validation and weight loading with file-size-based integrity validation to catch partial/corrupted downloads.
*   `face_restoration.py`: Wraps CodeFormer inference with cached device detection and a face-detection gate that returns the original image (with diagnostics) when no faces are found.
*   `refinement_pipeline.py`: Stage II SDXL-ControlNet pipeline with thread-safe loading, sketch pre-enhancement (CLAHE), `get_refinement_config()` integration, and configurable inference steps (default 40).
*   `sketch_refiner.py`: SDXL Img2Img editing (`StableDiffusionXLImg2ImgPipeline`) with post-generation mask compositing. Wires `prepare_enhanced_inpaint_inputs()` for feathering, auto-dilation, and adaptive strength. Fixes the UNet channel mismatch that previously caused masks to be silently ignored.
*   `visual_aids.py`: Inlines raw SVGs providing anatomical reference for operators.
*   `.streamlit/config.toml`: Enforces light mode and disables CORS/XSRF to support Cloudflare Tunnel deployments.
*   `inpaint_enhancements.py`: Fully integrated enhancement module providing mask feathering, graduated strength maps, prompt-based region inference, adaptive inpaint strength auto-tuning, and improved difference blending with soft attenuation curve for stable cascading edits.
*   `refinement_enhancements.py`: Fully integrated enhancement module providing adaptive ControlNet conditioning scale (fixed for non-binary fused edge maps), multi-modal edge fusion (Canny + Sobel + Laplacian), adaptive Canny threshold selection, CLAHE sketch pre-processing, and region-specific post-generation sharpening for eyes and mouth.

## License
Skaitch is licensed under the MIT License. See `LICENSE` for more information.

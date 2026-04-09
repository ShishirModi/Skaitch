# Skaitch
**Forensic Composite Suite · Generative Morphology Matching Pipeline**

Skaitch is a professional-grade forensic sketching and photorealistic refinement pipeline. It utilizes Stable Diffusion XL (SDXL) and ControlNet to translate categorical facial feature selections into high-fidelity investigative composites.

## 🌟 Core Features (V2.1+ Architecture)

*   **Phase I (Narrative Prompting):** Leverages a sophisticated prompt builder to construct grammatical narratives from categorical facial features. The SDXL Base 1.0 model parses this narrative to generate structural sketch variants in 4 styles: **Pencil sketch**, **Charcoal sketch**, **Police composite**, and **Forensic artist rendering**. Enhanced with ethnicity-specific anatomical grounding to ensure anatomically coherent outputs across diverse populations.
*   **Phase II (ControlNet Refinement):** Uses the `diffusers/controlnet-canny-sdxl-1.0` pipeline to elevate the finalized sketch into a photorealistic composite, maintaining precise geometric fidelity. Canny edge detection anchors the refinement to the sketch geometry with a configurable `controlnet_conditioning_scale`.
*   **Targeted Mask Inpainting:** Features the `streamlit-drawable-canvas` integration, allowing operators to draw masks and apply specific structural edits (e.g., "make the nose more pointed") without degrading the rest of the image. Optimized for reverse-proxy and Cloudflare Tunnel stability.
*   **Automated Face Restoration:** Non-destructive integration of the CodeFormer network (`sczhou/CodeFormer`) cleans up and sharpens generated faces, neutralizing diffusion artifacts before final rendering.
*   **Advanced Enhancement Modules (`inpaint_enhancements.py`, `refinement_enhancements.py`):** Drop-in modules providing mask feathering, graduated strength maps, automatic region inference from edit instructions, difference blending for cascading-edit stability, adaptive Canny thresholds, multi-modal edge fusion (Canny + Sobel + Laplacian), adaptive ControlNet conditioning scale, and region-specific post-processing sharpening for eyes and mouth.

## 🛠️ System Requirements

Skaitch is tailored for operation on standard cloud GPU instances with constrained VRAM. Memory offloading ensures stability.

*   **GPU:** Minimum 16GB VRAM (NVIDIA T4, RTX 3080, or equivalent).
*   **OS:** Linux (Ubuntu 22.04 LTS recommended), macOS Apple Silicon (M-series), or Windows (via WSL2).
*   **RAM:** 32GB System RAM.
*   **Storage:** 20GB+ fast NVMe SSD space for model weights.

## ⚙️ Setup & Installation

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

## 🚀 Execution

Launch the primary Streamlit interface:
```bash
streamlit run app.py
```
The application will be available at `http://localhost:8501`.

## 🎨 UI/UX Design System
Skaitch uses a tailored CSS overlay enforcing a clean, high-contrast B2B SaaS aesthetic. 
- **Light Mode Enforced:** Independent of browser or OS settings for uncompromising accuracy when evaluating visual aids.
- **Color Palette:** Charcoal (`#1C1C1E`) for text, Amber (`#F59E0B`) for action items, and Ghost White (`#FAFAFA`) for elevated card surfaces.

## 📁 Repository Structure

*   `app.py`: Streamlit orchestration, session state management, and UI logic.
*   `prompt_builder.py`: Maps UI dropdowns to narrative strings and applies SDXL syntax boosters. Includes `build_sdxl_forensic_prompt()`, `build_sdxl_refinement_prompt()`, `build_edit_prompt()` (for inpainting edits), ethnicity-specific anatomical grounding, adaptive guidance scaling, and prompt-based mask region inference.
*   `download_model.py`: Automates environment validation and weight loading.
*   `face_restoration.py`: Wraps CodeFormer inference alongside memory safety offloading.
*   `refinement_pipeline.py`: Stage II Canny edge detection and SDXL-ControlNet inferencing pipeline.
*   `sketch_refiner.py`: Localized SDXL Regional Inpainting (`StableDiffusionXLInpaintPipeline`) for iterative editing workflows.
*   `visual_aids.py`: Inlines raw SVGs providing anatomical reference for operators.
*   `.streamlit/config.toml`: Enforces light mode and disables CORS/XSRF to support Cloudflare Tunnel deployments.
*   `inpaint_enhancements.py`: Drop-in enhancement module providing mask feathering, graduated strength maps, prompt-based region inference, adaptive inpaint strength auto-tuning, and difference blending for stable cascading edits.
*   `refinement_enhancements.py`: Drop-in enhancement module providing adaptive ControlNet conditioning scale, multi-modal edge fusion (Canny + Sobel + Laplacian), adaptive Canny threshold selection, CLAHE sketch pre-processing, and region-specific post-generation sharpening for eyes and mouth.

## 🛡️ License
Skaitch is licensed under the MIT License. See `LICENSE` for more information.

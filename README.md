# Skaitch
**Forensic Composite Suite · Generative Morphology Matching Pipeline**

Skaitch is a professional-grade forensic sketching and photorealistic refinement pipeline. It utilizes Stable Diffusion XL (SDXL) and ControlNet to translate categorical facial feature selections into high-fidelity investigative composites.

## 🌟 Core Features (V2.1 Architecture)

*   **Phase I (Narrative Prompting):** Leverages a sophisticated prompt builder to construct grammatical narratives from categorical facial features. The SDXL Base 1.0 model parses this narrative to generate structural sketch variants.
*   **Phase II (ControlNet Refinement):** Uses the `diffusers/controlnet-canny-sdxl-1.0` pipeline to elevate the finalized sketch into a photorealistic composite, maintaining precise geometric fidelity.
*   **Targeted Mask Inpainting:** Features a custom React-based canvas (`skaitch_canvas`) integrated into Streamlit, allowing operators to draw masks and apply specific structural edits (e.g., "make the nose more pointed") without degrading the rest of the image.
*   **Automated Face Restoration:** Non-destructive integration of the CodeFormer network (`sczhou/CodeFormer`) cleans up and sharpens generated faces, neutralizing diffusion artifacts before final rendering.

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
*   `prompt_builder.py`: Maps UI dropdowns to narrative strings and applies SDXL syntax boosters.
*   `download_model.py`: Automates environment validation and weight loading.
*   `face_restoration.py`: Wraps CodeFormer inference alongside memory safety offloading.
*   `refinement_pipeline.py`: Stage II Canny edge detection and SDXL-ControlNet inferencing.
*   `sketch_refiner.py`: Localized SDXL Inpainting for iterative editing workflows.
*   `visual_aids.py`: Inlines raw SVGs providing anatomical reference for operators.
*   `skaitch_canvas/`: Custom bidirectional React component ensuring Streamlit state doesn't wipe active canvas masks.

## 🛡️ License
Skaitch is licensed under the MIT License. See `LICENSE` for more information.

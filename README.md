# Skaitch: A Two-Stage Generative Framework for High-Fidelity Forensic Composite Portraits

**Skaitch** is an advanced, GPU-accelerated application engineered to produce professional-grade forensic facial sketches from structured categorical inputs. Operating conceptually similarly to traditional police composite toolkits, Skaitch leverages modern Deep Generative Neural Networks in a dual-phase architecture.

Phase I translates semantic morphological descriptors into hyper-detailed pencil sketches via **Stable Diffusion XL (SDXL)**, further refined by **CodeFormer** face restoration geometry. Phase II maps the synthesized sketch into an **SDXL + ControlNet** refinement pass to achieve a photorealistic "fact-check" portrait translation through clinical edge-guidance.

![Skaitch Protocol](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![Torch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)

---

## 1. System Architecture Overview

Skaitch runs on a robust enterprise-grade Linux server architecture optimally designed for memory-intensive diffusion workflows. 

### Infrastructure Requirements
* **Compute:** NVIDIA T4 GPU (15GB VRAM), optimally supporting FP16 tensor precision.
* **Storage:** Fast NVMe SSDs (`/opt/dlami/nvme/models/`) utilized for caching massive diffusion UNet structures efficiently.
* **Environment:** Python 3.12 on Linux (Ubuntu recommended, given essential specific C-extensions needed for Jittor framework compilation).

---

## 2. Phase I: Generative Synthesis (SDXL + CodeFormer)

The first phase of the pipeline operates immediately upon the submission of semantic categorical parameters (e.g., *Jawline: Strong*, *Eyes: Almond*).

* **Model Loading:** The pipeline loads the `stabilityai/stable-diffusion-xl-base-1.0` model fully into VRAM (`device="cuda"`). Computations are strictly executed in `torch.float16` to circumvent OOM failures while employing SDXL native $1024 \times 1024$ latents.
* **Stochastic Variation:** Utilizing PyTorch pseudo-random generation across specific manual seeds, Skaitch simultaneously spawns three parallel latent diffusion processes. This ensures investigators are offered **3 varied conceptualizations** of the provided descriptions.
* **Face Restoration (CodeFormer):** Immediately following decoding via the VAE, raw SDXL image arrays are bridged to a localized instance of `sczhou/CodeFormer`. Functioning as a transformative blind-face restorer, CodeFormer recalculates the High-Frequency structural components of the facial topography, severely decreasing diffusion artifacts naturally propagating in sketch mediums.

---

## 3. Phase II: Photorealistic Refinement (SDXL + ControlNet)

The generation pipeline concludes with a high-fidelity refinement pass. Instead of a separate GAN, Skaitch leverages **SDXL ControlNet (Canny)** to translate the forensic sketch into a photorealistic portrait.

*   **Morphological Guidance:** The ControlNet module uses Canny edge detection on the generated sketch, ensuring the photorealistic output follows the source geometry with clinical precision.
*   **Hyper-Realistic Refinement:** A dedicated SDXL pass with a "Professional Studio Portrait" prompt synthesizes skin textures, lighting, and fine facial features at native 1024x1024 resolution.
*   **CodeFormer Integration:** All results undergo a final face restoration pass to ensure eye/mouth symmetry and remove potential diffusion artifacts.

---

## 4. Technical Evolution & Legacy Analysis

Skaitch has transitioned through two major architectural phases for Photorealistic Translation.

### 4.1 Legend: The Jittor/DFD Era (Legacy)
Originally, Phase II relied on the **DeepFaceDrawing (DFD)** GAN framework using the Jittor compiler.
- **Advantages:** Specialized manifold learning for "fixing" human sketches.
- **Disadvantages:** Significant compilation overhead on modern Linux (GCC 13+ conflicts), rigid 512x512 output, and complex weight management (DVC/Baidu Pan dependencies).

### 4.2 Modern: SDXL-ControlNet (Current)
The current architecture utilizes a unified Diffusers/PyTorch stack.
- **Advantages:** 
    - **Stability:** Removes all JIT compilation and C++ compiler friction.
    - **Resolution:** Native 1024x1024 output vs 512x512 legacy.
    - **Maintenance:** Uses standard `.safetensors` weights via HuggingFace Hub.
    - **Quality:** Higher dynamic range and superior skin texture synthesis.
- **Disadvantages:** Higher VRAM requirement (~16GB+ recommended for peak throughput), though optimized for T4 via CPU offloading.

---

## 5. Setup & Deployment Guidelines

### I. Repository Clone & Requirements
```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
# Install modern dependencies (Torch 2.3+, Diffusers, Transformers)
pip install -r requirements.txt
```

### II. Automated Weights Setup
Skaitch handles the entire setup of both Phase I and Phase II automatically. Simply run the application:
```bash
streamlit run app.py
```

**What happens during initialization:**
1.  **SDXL Base:** Downloaded to `/opt/dlami/nvme/models/sdxl/`.
2.  **ControlNet Canny:** Downloaded to `/opt/dlami/nvme/models/controlnet-canny-sdxl/`.
3.  **CodeFormer:** Fetched from GitHub Releases to `/opt/dlami/nvme/models/codeformer/`.
4.  **No Manual Intervention:** All weights are sourced automatically from validated mirrors.

---

## 6. User Interface (Streamlit)

Skaitch abstracts complex PyTorch interactions beneath an intuitive Streamlit browser application. 
- **Telemetry Overview:** The sidebar reports real-time CUDA properties (Model Name, Available VRAM).
- **Control Interface:** Options intelligently toggle based on "Free-Text" vs "Forensic Sketch Mode".
- **Auto-Persistent Save Architecture:** Generated variants and resulting Refinement translations are locally cached under `data/`.

# Skaitch: A Two-Stage Generative Framework for High-Fidelity Forensic Composite Portraits

**Skaitch** is an advanced, GPU-accelerated application engineered to produce professional-grade forensic facial sketches from structured categorical inputs. Operating conceptually similarly to traditional police composite toolkits, Skaitch leverages modern Deep Generative Neural Networks in a dual-phase architecture.

Phase I translates semantic morphological descriptors into hyper-detailed pencil sketches via **Stable Diffusion XL (SDXL)**, further refined by **CodeFormer** face restoration geometry. Phase II maps the synthesized sketch into the **DeepFaceDrawing** conditional GAN to achieve a photorealistic "fact-check" portrait translation through intermediate manifold learning.

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

## 3. Phase II: Photorealistic Translation (DeepFaceDrawing)

The highest fidelity sketch is piped implicitly into Phase II, relying heavily on geometry and boundary preservation translation mapping natively conceptualized by the researchers of DeepFaceDrawing (Chen et al.).

* **Semantic Manifold Projection:** The DeepFaceDrawing model intrinsically contains three overarching operational modules:
  * **Component Embedding (CE):** Five distinct auto-encoders linearly encapsulate vectors specific to eyes, nose, mouth, and structural silhouette.
  * **Feature Mapping (FM):** Synthesizes isolated local manifolds into spatial probability fields.
  * **Image Synthesis (IS):** A pix2pix conditional GAN structure conditionally generating RGB realistic interpolations matching the sketch shadows.
* **Jittor AI Framework:** Phase II relies on **Jittor**, a dynamic deep learning compilation framework native to Linux. We have programmatically unified the application context so Jittor will forcefully inherit GPU execution allocations alongside PyTorch natively (`jt.flags.use_cuda = 1`), drastically dropping phase translation from minutes to sub-second runtimes.

---

## 4. Setup & Deployment Guidelines

### I. Repository Clone & Requirements
```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
# Install dependencies securely inside a clean Python 3.12 setup
pip install -r requirements.txt
```
*Note on Jittor:* Ensure you install your distribution's Python development headers (`python3-dev`) prior to executing `pip install jittor`. 

### II. Parameter Configuration
Environment secrets (Admin portal parameters) and automated DeepFaceDrawing mirror locations are structured in the root configuration:
```bash
cp .env.example .env
nano .env # Assign your unique ADMIN_PASSWORD
```

### III. Automated Weights Loading
Instead of maintaining fragmented model caches, weights are centralized. Simply boot the application:
```bash
streamlit run app.py
```
*During initialization, Skaitch will traverse the configured NVMe drive (`/opt/dlami/nvme/models/`). Failing validation, Skaitch will spawn a direct download conduit via the HuggingFace CLI natively saving 10GB+ optimized weights bypassing local memory bottlenecks.*

**Action Required for DeepFaceDrawing:**
The required DeepFaceDrawing `.pkl` checkpoints are securely hosted behind a Baidu Pan token access screen. Before full functionality occurs, ensure you navigate your `download_model.py` CLI instructions:
1. Access the provided Baidu drive.
2. Download and drop the checkpoint `.pkl` files within `external/DeepFaceDrawing/Params/` (Or supply a secure mirror endpoint uniformly within `DFD_WEIGHTS_DIRECT_URL`).

---

## 5. User Interface (Streamlit)

Skaitch abstracts complex PyTorch interactions beneath an intuitive Streamlit browser application. 
- **Telemetry Overview:** The sidebar reports real-time CUDA properties (Model Name, Available VRAM).
- **Control Interface:** Options intelligently toggle based on "Free-Text" vs "Forensic Sketch Mode" ensuring conditional generation paths perfectly mimic the pipeline's operational boundaries.
- **Auto-Persistent Save Architecture:** Generated variants and resulting DeepFaceDrawing translation checks are locally cached under `data/` tagged chronologically alongside active metadata strings, guaranteeing total retention unless specifically disabled via the secure *Admin Settings* module.

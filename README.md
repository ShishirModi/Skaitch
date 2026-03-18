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

### 3. Phase II: Photorealistic Translation (DeepFaceDrawing)

The highest fidelity sketch is piped into Phase II, relying on the **PyTorch implementation** of DeepFaceDrawing (Xu-Justin version). This phase translates the sketch's morphological boundaries into a photorealistic manifold.

*   **Semantic Manifold Projection:** The architecture consists of three core modules:
    *   **Component Embedding (CE):** Encapsulates feature vectors for eyes, nose, mouth, and face silhouette.
    *   **Feature Mapping (FM):** Map these embeddings into spatial feature maps.
    *   **Image Synthesis (IS):sibA GAN-based generator that synthesizes the final photorealistic image from the feature maps.
*   **Performance Optimization:** By utilizing native PyTorch, we eliminate the previous Just-In-Time (JIT) compilation bottlenecks found in Jittor. This allows for **sub-second inference** on the NVIDIA T4 GPU immediately upon the first generation.

---

## 4. Setup & Deployment Guidelines

### I. Repository Clone & Requirements
```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
# Install modern dependencies (Torch 2.3+, Diffusers, Transformers)
pip install -r requirements.txt
```

### II. Automated Weights & Source Setup
Skaitch now handles the entire setup of Phase II automatically. Simply run the application:
```bash
streamlit run app.py
```

**What happens during initialization:**
1.  **SDXL & CodeFormer:** Downloaded to `/opt/dlami/nvme/models/` via HuggingFace and GitHub Releases.
2.  **Phase II (PyTorch DFD):** The application will automatically clone the DeepFaceDrawing PyTorch repository and its 1GB+ weights directly into `external/DeepFaceDrawing/`.
3.  **No Manual Intervention:** Unlike previous versions, you no longer need to manually download weights from Baidu Pan or Google Drive. Everything is sourced from GitHub mirrors.

---

## 5. User Interface (Streamlit)

Skaitch abstracts complex PyTorch interactions beneath an intuitive Streamlit browser application. 
- **Telemetry Overview:** The sidebar reports real-time CUDA properties (Model Name, Available VRAM).
- **Control Interface:** Options intelligently toggle based on "Free-Text" vs "Forensic Sketch Mode".
- **Auto-Persistent Save Architecture:** Generated variants and resulting DeepFaceDrawing translations are locally cached under `data/`, guaranteeing total retention for investigative review.

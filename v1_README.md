# Skaitch: A Two-Stage Generative Framework for High-Fidelity Forensic Composite Portraits

**Skaitch** is a specialized, GPU-accelerated forensic application engineered to produce professional-grade composite portraits from structured categorical inputs. By bridging expert morphological descriptors with a dual-phase generative pipeline, Skaitch transforms verbal descriptions into high-fidelity visual evidence.

Phase I translates semantic descriptors into hyper-detailed pencil sketches via **Stable Diffusion XL (SDXL)**, immediately refined by **CodeFormer** face restoration. Phase II leverages the synthesized sketch as a geometric anchor for an **SDXL-ControlNet** refinement pass, achieving a photorealistic "photographic" translation.

![Skaitch Protocol](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![Torch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)

---

## 1. System Architecture Overview

Skaitch is a unified PyTorch-based framework designed for high-fidelity portrait synthesis on memory-constrained hardware (e.g., NVIDIA T4).

### Infrastructure Requirements
* **Compute:** NVIDIA T4 GPU (15GB VRAM), utilizing **Model CPU Offloading** via `accelerate`.
* **Storage:** Fast NVMe SSDs (`/opt/dlami/nvme/models/`) for caching SDXL and ControlNet weights.
* **Environment:** Python 3.12+ (Linux recommended). No JIT or C++ compilation required.

---

## 2. Phase I: Sketch Synthesis (SDXL + CodeFormer)

Phase I translates semantic morphological descriptors into a high-detail pencil sketch.

* **VRAM Optimization:** The pipeline uses `enable_model_cpu_offload()` to manage weight residency, enabling SDXL inference on a single T4. Computations are executed in `torch.float16` at $1024 \times 1024$ resolution.
* **Stochastic Variation:** Skaitch generates **3 parallel variants** per run, providing investigators with alternative interpretations of the same description.
* **Face Restoration:** Raw sketches are processed by **CodeFormer**, which sharpens facial geometry and restores structural integrity to translated descriptors.

---

## 3. Phase II: Photorealistic Refinement (SDXL-ControlNet)

The refinement phase transforms the forensic sketch into a photographic-quality evidence portrait.

* **Dynamic Feature Synchronization:** Feature selections (Hair Color, Eyes, Skin Tone, etc.) are propagated from Phase I to Phase II. This ensures the photorealistic result honors the specific traits selected by the investigator.
* **Dynamic Feature Synchronization:** Feature selections (Ethnicity, Eye Color, Hair, etc.) are propagated from Phase I to Phase II. This ensures the photorealistic result honors the specific traits selected by the investigator.
* **Morphological Guidance:** Uses **SDXL-ControlNet (Canny)** with an optimized **0.70 conditioning scale** to anchor the photographic synthesis to the sketch's geometry.
* **Refinement Precision:** Implements **Weighted Prompting** (e.g., 1.4x for eyes, 1.3x for hair/glasses) to prevent the source sketch from overriding color and accessory instructions.
* **Comprehensive Morphology:** Supports a global ethnicity library (East Asian, South Asian, etc.) and specialized accessories like frame-shaped spectacles with adjustable tints.

---

## 4. Technical Evolution: From GANs to Diffusion

Skaitch has been re-engineered to move past legacy limitations.

### 4.1 Legacy: The DFD/Jittor Era
Originally, Phase II used **DeepFaceDrawing (DFD)** on the Jittor framework.
- **Issues:** Rigid 512x512 resolution, extreme dependency friction (JVC/GCC-13), and limited realism.

### 4.2 Modern: SDXL-ControlNet (Current)
The current architecture is built on a modern, stable Diffusers stack.
- **Superior Quality:** Native 1024px resolution and cinematic skin synthesis.
- **Stability:** Removed all custom JIT compilers, making the app portable across standard Linux/CUDA environments.
- **Precision:** Dynamic prompt synchronization ensures the "Photographic" pass matches the "Sketch" pass exactly.

---

## 5. Deployment & Setup

### I. Installation
```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
pip install -r requirements.txt
```

### II. Automated Weights Initialization
Skaitch automatically manages all weights on the NVMe drive. Simply run:
```bash
streamlit run app.py
```

**Automated Setup Includes:**
- **SDXL Base:** `/opt/dlami/nvme/models/sdxl/`
- **ControlNet Canny:** `/opt/dlami/nvme/models/controlnet-canny-sdxl/`
- **CodeFormer:** `/opt/dlami/nvme/models/codeformer/` (Weights) and `external/CodeFormer/` (Repo)

---

## 6. User Interface

- **Forensic Specialty UI:** A focused, structured workstation. The legacy free-text "Free Form" mode has been removed to minimize operator error.
- **Telemetry Overview:** Real-time monitoring of CUDA device and VRAM status in the sidebar.
- **Auto-Persistent Storage:** All sketches and refinements are automatically saved to the local `data/` directory with unique timestamps.


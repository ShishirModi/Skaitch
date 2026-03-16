# Skaitch

A local **Stable Diffusion** image generation tool with a polished **Streamlit** frontend. Type a prompt, tweak the settings, and generate images entirely on your machine — no cloud API needed.

![Skaitch UI](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

---

## Features

| Feature | Description |
|---|---|
| **Text-to-image** | Generate images from natural-language prompts using Stable Diffusion v1.5 |
| **Forensic Sketch Mode** | Build police-grade composite sketches by selecting facial features from dropdowns |
| **DeepFaceDrawing Fact-Check** | Convert generated sketches into photorealistic images dynamically using a secondary Jittor model pipeline |
| **Negative prompts** | Specify what you *don't* want to see in the output |
| **Parameter control** | Tune inference steps, guidance scale, dimensions, and seed |
| **Reproducibility** | Fix the seed to regenerate the exact same image |
| **One-click download** | Save the generated image as PNG |
| **Cached model** | The pipeline loads once and stays in memory across reruns |

---

## Project Structure

```
Skaitch/
├── app.py                  # Streamlit frontend
├── prompt_builder.py       # Forensic sketch prompt assembly module
├── scripts/
│   └── sd_test.py          # Headless CLI test script
├── external/
│   └── stable_diffusion/   # Local model weights (SD v1.5)
├── data/                   # Generated outputs (git-ignored)
├── requirements.txt
└── skaitch_env/            # Python virtual environment
```

---

## Prerequisites

- **Python 3.10+**
- **Linux Environment** (Required for compiling the Jittor neural network backend powering DeepFaceDrawing)
- **Git** (to clone the repo)
- **Stable Diffusion v1.5 weights** placed in `external/stable_diffusion/`. You can download them from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) or use the `huggingface-cli`:
  ```bash
  huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir external/stable_diffusion
  ```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShishirModi/Skaitch.git
cd Skaitch
```

### 2. Create a virtual environment & install dependencies

```bash
python3 -m venv skaitch_env
source skaitch_env/bin/activate
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## Usage

### Free-Text Mode (default)

1. **Enter a prompt** in the sidebar (e.g. *"a futuristic cityscape at sunset, digital art"*).
2. *(Optional)* Add a **negative prompt** to exclude unwanted elements.
3. Adjust **Inference steps**, **Guidance scale**, **Width/Height**, and **Seed**.
4. Click **🚀 Generate**.
5. Once complete, view the result and click **⬇️ Download PNG** to save.

### Forensic Sketch Mode

1. Toggle **🔍 Forensic Sketch Mode** on in the sidebar.
2. Select facial features from the dropdowns:

   | Category | Examples |
   |---|---|
   | Gender / Age | Male, 26–35 |
   | Face shape / Jawline | Oval, Strong |
   | Eyes / Eyebrows | Almond, Thick |
   | Nose / Mouth | Straight, Thin |
   | Skin tone | Medium |
   | Hair style / color | Short cropped, Dark brown |
   | Facial hair | Stubble |
   | Distinguishing marks | Scar on left cheek |

3. Choose a **Sketch style** (Pencil sketch, Charcoal sketch, Police composite, or Forensic artist rendering).
4. *(Optional)* Add **Additional details** as free text.
5. Review the **Generated prompt** preview, then click **🚀 Generate**.

> Forensic mode auto-sets guidance scale to **10.0** and inference steps to **30** for better sketch quality. You can still adjust these manually.

> **DeepFaceDrawing Integration:** When Forensic Mode is activated, the generated sketch is automatically processed via our secondary **DeepFaceDrawing (Jittor)** pipeline. The model parses your morphological feature selections (like Jawline width) into mathematical tensors, adjusting the model weights to convert the sketch into an accurate, photorealistic portrait displayed side-by-side with the sketch.

> **Note:** Generation runs on CPU by default and can take several minutes. A CUDA-capable GPU will significantly speed things up.

---

## Headless / CLI Usage

For quick tests without the UI, use the standalone script:

```bash
source skaitch_env/bin/activate
python scripts/sd_test.py
```

This generates a forensic sketch with sample features to `data/test_output.png`.

---

## Configuration

All generation parameters are exposed in the Streamlit sidebar:

| Parameter | Range | Default (Free) | Default (Forensic) | Effect |
|---|---|---|---|---|
| Inference steps | 1 – 50 | 20 | 30 | More steps → finer details |
| Guidance scale | 1.0 – 20.0 | 7.5 | 10.0 | Higher → closer to prompt |
| Width | 256, 512 | 512 | 512 | Output width in pixels |
| Height | 256, 512 | 512 | 512 | Output height in pixels |
| Seed | 0 – 2³² | 0 (random) | 0 (random) | Fix for reproducibility |

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — interactive web frontend
- **[Hugging Face Diffusers](https://huggingface.co/docs/diffusers)** — Stable Diffusion pipeline
- **[PyTorch](https://pytorch.org/)** — tensor computation & model inference
- **[Pillow](https://python-pillow.org/)** — image handling

---

## License

This project is for personal / educational use. Stable Diffusion model weights are subject to the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).

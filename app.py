# File: app.py
# Purpose: Streamlit frontend for Stable Diffusion image generation

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import os

from prompt_builder import (
    FACIAL_FEATURES,
    SKETCH_STYLES,
    FORENSIC_DEFAULTS,
    build_forensic_prompt,
)

from visual_aids import VISUAL_AIDS, get_svg_html
import dfd_integration

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skaitch – Stable Diffusion",
    page_icon="🎨",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Typography ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Global tweaks ───────────────────────────────────────────────── */
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.4rem 2.8rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        color: #fff;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(168,85,247,.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.6;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1f 0%, #161636 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* section headers in sidebar */
    .sidebar-section {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-section .icon {
        font-size: 1.1rem;
    }
    .sidebar-section .label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(224,224,255,0.55);
    }

    /* sidebar title */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e8e8ff;
        margin-bottom: 0.2rem;
        letter-spacing: -0.3px;
    }
    .sidebar-subtitle {
        font-size: 0.78rem;
        color: rgba(200,200,255,0.4);
        margin-bottom: 1rem;
        font-weight: 300;
    }

    /* ── Generate button ─────────────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6c5ce7, #a855f7);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 14px;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(108, 92, 231, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(108, 92, 231, 0.45);
        background: linear-gradient(135deg, #7c6cf7, #b865ff);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Parameter chip cards ────────────────────────────────────────── */
    .param-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(8px);
    }
    .param-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .param-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.88rem;
    }
    .param-row:last-child { border-bottom: none; }
    .param-row .key {
        color: rgba(255,255,255,0.5);
        font-weight: 400;
    }
    .param-row .val {
        color: #e8e8ff;
        font-weight: 600;
    }

    /* ── Prompt preview ─────────────────────────────────────────────── */
    .prompt-preview {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: rgba(200,200,255,0.7);
        line-height: 1.55;
        margin-top: 0.5rem;
        max-height: 140px;
        overflow-y: auto;
        word-break: break-word;
    }
    .prompt-preview-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(167,139,250,0.7);
        margin-top: 0.8rem;
        margin-bottom: 0.2rem;
    }

    /* ── Empty-state hero ────────────────────────────────────────────── */
    .empty-hero {
        text-align: center;
        padding: 5rem 2rem;
        color: rgba(255,255,255,0.45);
    }
    .empty-hero .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    .empty-hero h3 {
        font-size: 1.3rem;
        font-weight: 600;
        color: rgba(255,255,255,0.65);
        margin: 0 0 0.4rem 0;
    }
    .empty-hero p {
        font-size: 0.92rem;
        max-width: 380px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Success banner ──────────────────────────────────────────────── */
    .success-banner {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(52,211,153,0.08));
        border: 1px solid rgba(16,185,129,0.2);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .success-banner .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #10b981;
        flex-shrink: 0;
    }
    .success-banner span {
        color: #6ee7b7;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* ── Download button override ─────────────────────────────────────── */
    .stDownloadButton > button {
        width: 100%;
        background: rgba(255,255,255,0.06);
        color: #c4b5fd;
        border: 1px solid rgba(196,181,253,0.2);
        padding: 0.7rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        border-radius: 12px;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(196,181,253,0.12);
        border-color: rgba(196,181,253,0.35);
        color: #e2d9fe;
    }

    /* ── Image container ─────────────────────────────────────────────── */
    .stImage {
        border-radius: 16px;
        overflow: hidden;
    }
    .stImage img {
        border-radius: 16px;
    }

    /* ── Hide default Streamlit branding ─────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "external", "stable_diffusion")


# ─── Cached model loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Stable Diffusion model …")
def load_pipeline(model_path: str):
    """Load the Stable Diffusion pipeline once and cache across reruns."""
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe = pipe.to("cpu")
    return pipe


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🎨 Skaitch — Stable Diffusion</h1>
        <p>Generate images from text prompts using a local Stable Diffusion model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Skaitch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Stable Diffusion · Local</div>',
        unsafe_allow_html=True,
    )

    # ── Mode toggle ────────────────────────────────────────────────────────
    forensic_mode = st.toggle("🔍 Forensic Sketch Mode", value=False)

    if forensic_mode:
        # ── Section: Facial Features ───────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">'
            '<span class="icon">👤</span>'
            '<span class="label">Facial Features</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        selected_features: dict[str, str] = {}

        # Layout: Gender + Age side-by-side
        col_g, col_a = st.columns(2)
        with col_g:
            selected_features["Gender"] = st.selectbox(
                "Gender", FACIAL_FEATURES["Gender"]
            )
        with col_a:
            selected_features["Age range"] = st.selectbox(
                "Age range", FACIAL_FEATURES["Age range"], index=1
            )

        # Face structure row
        col_fs, col_jl = st.columns(2)
        with col_fs:
            feature_val = st.selectbox("Face shape", FACIAL_FEATURES["Face shape"])
            selected_features["Face shape"] = feature_val
            st.markdown(get_svg_html(VISUAL_AIDS["Face shape"][feature_val]), unsafe_allow_html=True)
            
        with col_jl:
            feature_val = st.selectbox("Jawline", FACIAL_FEATURES["Jawline"])
            selected_features["Jawline"] = feature_val
            st.markdown(get_svg_html(VISUAL_AIDS["Jawline"][feature_val]), unsafe_allow_html=True)

        # Eyes + Brows row
        col_e, col_eb = st.columns(2)
        with col_e:
            feature_val = st.selectbox("Eyes", FACIAL_FEATURES["Eyes"])
            selected_features["Eyes"] = feature_val
            st.markdown(get_svg_html(VISUAL_AIDS["Eyes"][feature_val]), unsafe_allow_html=True)
            
        with col_eb:
            selected_features["Eyebrows"] = st.selectbox(
                "Eyebrows", FACIAL_FEATURES["Eyebrows"]
            )

        # Nose + Mouth row
        col_n, col_m = st.columns(2)
        with col_n:
            feature_val = st.selectbox("Nose", FACIAL_FEATURES["Nose"])
            selected_features["Nose"] = feature_val
            st.markdown(get_svg_html(VISUAL_AIDS["Nose"][feature_val]), unsafe_allow_html=True)
            
        with col_m:
            selected_features["Mouth / Lips"] = st.selectbox(
                "Mouth / Lips", FACIAL_FEATURES["Mouth / Lips"]
            )

        # Skin tone
        selected_features["Skin tone"] = st.selectbox(
            "Skin tone", FACIAL_FEATURES["Skin tone"], index=2
        )

        # ── Section: Hair ──────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">'
            '<span class="icon">💇</span>'
            '<span class="label">Hair</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        col_hs, col_hc = st.columns(2)
        with col_hs:
            selected_features["Hair style"] = st.selectbox(
                "Hair style", FACIAL_FEATURES["Hair style"]
            )
        with col_hc:
            selected_features["Hair color"] = st.selectbox(
                "Hair color", FACIAL_FEATURES["Hair color"]
            )

        selected_features["Facial hair"] = st.selectbox(
            "Facial hair", FACIAL_FEATURES["Facial hair"]
        )

        # ── Section: Distinguishing Marks ──────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">'
            '<span class="icon">🔖</span>'
            '<span class="label">Distinguishing Marks</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        selected_features["Distinguishing marks"] = st.selectbox(
            "Distinguishing marks",
            FACIAL_FEATURES["Distinguishing marks"],
            label_visibility="collapsed",
        )

        # ── Sketch style ──────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-section">'
            '<span class="icon">🖊️</span>'
            '<span class="label">Sketch Style</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        sketch_style = st.selectbox(
            "Sketch style", SKETCH_STYLES, label_visibility="collapsed"
        )

        extra_details = st.text_input(
            "Additional details",
            placeholder="e.g. wearing glasses, prominent ears …",
            help="Free-text appended to the generated prompt.",
        )

        # Build prompt
        prompt, negative_prompt = build_forensic_prompt(
            selected_features, sketch_style, extra_details
        )

        # Show prompt preview
        st.markdown(
            '<div class="prompt-preview-label">Generated prompt</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="prompt-preview">{prompt}</div>',
            unsafe_allow_html=True,
        )

        # Use forensic defaults
        default_steps = FORENSIC_DEFAULTS["num_inference_steps"]
        default_cfg = FORENSIC_DEFAULTS["guidance_scale"]

    else:
        # ── Section: Prompt (free-text mode) ───────────────────────────────
        st.markdown(
            '<div class="sidebar-section">'
            '<span class="icon">✏️</span>'
            '<span class="label">Prompt</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        prompt = st.text_area(
            "Prompt",
            value="close up shot of a pink lotus flower in the center, photorealistic, high detail",
            height=110,
            help="Describe the image you want to generate.",
            label_visibility="collapsed",
        )

        negative_prompt = st.text_area(
            "Negative prompt",
            value="",
            height=70,
            placeholder="(optional) blurry, low quality, distorted …",
            help="Describe what you do NOT want in the image.",
            label_visibility="collapsed",
        )

        default_steps = 20
        default_cfg = 7.5

    # ── Section: Parameters ────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<span class="icon">🎛️</span>'
        '<span class="label">Parameters</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    num_inference_steps = st.slider(
        "Inference steps",
        min_value=1,
        max_value=50,
        value=default_steps,
        help="More steps → higher quality but slower generation.",
    )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=1.0,
        max_value=20.0,
        value=default_cfg,
        step=0.5,
        help="How closely to follow the prompt. Higher = more faithful, lower = more creative.",
    )

    # ── Section: Dimensions ────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<span class="icon">📐</span>'
        '<span class="label">Dimensions</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_w, col_h = st.columns(2)
    with col_w:
        width = st.selectbox("Width", [256, 512], index=1)
    with col_h:
        height = st.selectbox("Height", [256, 512], index=1)

    seed = st.number_input(
        "Seed (0 = random)",
        min_value=0,
        max_value=2**32 - 1,
        value=0,
        help="Set a fixed seed to reproduce the same result.",
    )

    # ── Generate ───────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    generate = st.button("🚀  Generate", use_container_width=True)

# ─── Generation logic ─────────────────────────────────────────────────────────
if generate:
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating.")
        st.stop()

    pipe = load_pipeline(MODEL_PATH)

    # Build a generator for reproducibility if seed != 0
    generator = None
    if seed != 0:
        generator = torch.Generator("cpu").manual_seed(int(seed))

    with st.spinner("🖌️ Generating — this may take a few minutes on CPU …"):
        result = pipe(
            prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image: Image.Image = result.images[0]

    # ── Success banner ─────────────────────────────────────────────────
    st.markdown(
        '<div class="success-banner">'
        '<div class="dot"></div>'
        "<span>Image generated successfully</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Pipeline specific displays ──────────────────────────────────────
    if forensic_mode:
        with st.spinner("🤖 DeepFaceDrawing Fact-Check — Generating Photorealistic Output …"):
            try:
                dfd_image = dfd_integration.run_dfd(image, selected_features)
                dfd_success = True
            except Exception as e:
                dfd_error = str(e)
                dfd_success = False

        col_img1, col_img2, col_meta = st.columns([1.5, 1.5, 1], gap="medium")
        
        with col_img1:
            st.markdown("#### Stable Diffusion Sketch")
            st.image(image, use_container_width=True)
            st.caption(f"*\"{prompt}\"*")

        with col_img2:
            st.markdown("#### Photorealistic Fact-Check")
            if dfd_success:
                st.image(dfd_image, use_container_width=True)
                st.caption(f"*DeepFaceDrawing (Jittor)*")
            else:
                st.error(f"DeepFaceDrawing failed: {dfd_error}\nEnsure you are on a compatible Linux environment with Jittor compiled.")

    else:
        # Standard layout for free-text mode
        col_img1, col_img2, col_meta = st.columns([0, 3, 1], gap="large")
        with col_img2:
            st.image(image, use_container_width=True)
            st.caption(f"*\"{prompt}\"*")

    # ── Meta info (Right Column) ──────────────────────────────────────────
    with col_meta:
        seed_row = ""
        if seed != 0:
            seed_row = (
                '<div class="param-row">'
                '<span class="key">Seed</span>'
                f'<span class="val">{seed}</span>'
                "</div>"
            )
        mode_row = ""
        if forensic_mode:
            mode_row = (
                '<div class="param-row">'
                '<span class="key">Mode</span>'
                '<span class="val">Forensic</span>'
                "</div>"
            )
        st.markdown(
            '<div class="param-card">'
            "<h4>Parameters</h4>"
            f"{mode_row}"
            '<div class="param-row">'
            '<span class="key">Steps</span>'
            f'<span class="val">{num_inference_steps}</span>'
            "</div>"
            '<div class="param-row">'
            '<span class="key">CFG</span>'
            f'<span class="val">{guidance_scale}</span>'
            "</div>"
            '<div class="param-row">'
            '<span class="key">Size</span>'
            f'<span class="val">{width}×{height}</span>'
            "</div>"
            f"{seed_row}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button(
            label="⬇️  Download Sketch (PNG)",
            data=buf.getvalue(),
            file_name="skaitch_sketch.png",
            mime="image/png",
            use_container_width=True,
        )
        
        if forensic_mode and dfd_success:
            st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
            buf2 = io.BytesIO()
            dfd_image.save(buf2, format="PNG")
            st.download_button(
                label="⬇️  Download Fact-Check (PNG)",
                data=buf2.getvalue(),
                file_name="skaitch_photorealistic_factcheck.png",
                mime="image/png",
                use_container_width=True,
            )

else:
    # ── Empty-state hero ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="empty-hero">
            <div class="icon">🖼️</div>
            <h3>No image yet</h3>
            <p>Configure your prompt and settings in the sidebar, then press
            <strong>Generate</strong> to create an image.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

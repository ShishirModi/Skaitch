# File: app.py
# Purpose: Streamlit frontend for Stable Diffusion image generation

import streamlit as st

# Monkey-patch to fix streamlit-drawable-canvas compatibility with modern Streamlit (>=1.30.0)
# which removed the internal st_image.image_to_url function.
import streamlit.elements.image as st_image
if not hasattr(st_image, "image_to_url"):
    import base64
    import io
    def _monkeypatch_image_to_url(image, width, clamp, channels, output_format, image_id):
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=output_format)
        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return f"data:image/{output_format.lower()};base64,{img_str}"
    
    st_image.image_to_url = _monkeypatch_image_to_url

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import io
import os
from dotenv import load_dotenv

import download_model

# Load environment variables (e.g., ADMIN_PASSWORD)
load_dotenv()

from prompt_builder import (
    FACIAL_FEATURES,
    SKETCH_STYLES,
    FORENSIC_DEFAULTS,
    build_sdxl_forensic_prompt,
)

from visual_aids import VISUAL_AIDS, get_svg_html
import refinement_pipeline
import face_restoration

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skaitch – Forensic Composite Suite",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Typography & Base Theme ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Accent colour justification: Forensic Cyan (#06b6d4) provides a cold, 
       clinical precision fitting for a law enforcement intelligence dashboard,
       standing out against the Slate dark mode without feeling overly flashy. */

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #020617; /* Very dark slate */
    }

    /* Restore Streamlit icon fonts */
    [class*="material-symbols"],
    [data-testid*="Icon"],
    [data-testid*="stIcon"],
    .st-icon {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }

    /* ── Global Layout tweaks ────────────────────────────────────────── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1300px;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .main-header {
        background: #0f172a; /* Slate 900 */
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: clamp(1.2rem, 4vw, 1.8rem) clamp(1.5rem, 4vw, 2.2rem);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: clamp(1.4rem, 3vw, 1.8rem);
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    .sidebar-section {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .sidebar-section .icon {
        font-size: 1rem;
        opacity: 0.8;
    }
    .sidebar-section .label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #94a3b8;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.1rem;
        letter-spacing: -0.4px;
    }
    .sidebar-subtitle {
        font-size: 0.7rem;
        color: #06b6d4; /* Forensic Cyan */
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Main Canvas Feature Cards ───────────────────────────────────── */
    .feature-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .feature-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #06b6d4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        padding-bottom: 0.6rem;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 0.6rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #1e293b;
        color: #e2e8f0;
    }
    .stButton > button:hover {
        background: #334155;
        border-color: rgba(255, 255, 255, 0.2);
        color: #f8fafc;
    }
    /* Primary Action Buttons */
    div[data-testid="stButton"] button[kind="primary"] {
        background: #06b6d4 !important;
        color: #082f49 !important; /* Cyan 900 text */
        border: none !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.2) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #0891b2 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.3) !important;
    }

    /* ── Variant Cards (Sketch Output) ───────────────────────────────── */
    .variant-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.2), 0 4px 15px rgba(0,0,0,0.3);
        transition: border-color 0.2s ease;
    }
    .variant-card:hover {
        border-color: rgba(6, 182, 212, 0.4);
    }

    /* ── Parameter Chips / Metadata Table ────────────────────────────── */
    .param-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    .param-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .param-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }
    .param-row:last-child { border-bottom: none; }
    .param-row .key {
        color: #64748b;
    }
    .param-row .val {
        color: #06b6d4;
        font-weight: 500;
    }

    /* ── Prompt preview ──────────────────────────────────────────────── */
    .prompt-preview {
        background: #020617;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-top: 0.5rem;
        max-height: 120px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace;
    }
    .prompt-preview-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    /* ── Success banner ──────────────────────────────────────────────── */
    .success-banner {
        background: rgba(6, 182, 212, 0.05);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .success-banner .dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #06b6d4;
        flex-shrink: 0;
        box-shadow: 0 0 8px rgba(6, 182, 212, 0.6);
    }
    .success-banner span {
        color: #e2e8f0;
        font-size: 0.88rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* ── Image container ─────────────────────────────────────────────── */
    .stImage img {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Phase II Featured Image */
    .featured-output img {
        border-radius: 12px;
        border: 1px solid rgba(6, 182, 212, 0.3);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    /* ── Download button override ─────────────────────────────────────── */
    .stDownloadButton > button {
        background: transparent;
        color: #06b6d4;
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(6, 182, 212, 0.08);
        border-color: rgba(6, 182, 212, 0.5);
        color: #22d3ee;
    }

    /* ── Hide Streamlit branding ─────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "/opt/dlami/nvme/models/sdxl"
CODEFORMER_PATH = "/opt/dlami/nvme/models/codeformer"

# ─── Cached setup & loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Verifying models on NVMe... (may take minutes on first run)")
def ensure_models_exist():
    download_model.check_and_download_models()
    return True

@st.cache_resource(show_spinner="Loading SDXL model into VRAM …")
def load_pipeline():
    ensure_models_exist()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    )
    if torch.cuda.is_available():
        # Use model-level CPU offloading to save VRAM on T4
        pipe.enable_model_cpu_offload()
    return pipe

def run_codeformer(img: Image.Image) -> Image.Image:
    """True CodeFormer face restoration. Returns original on failure."""
    try:
        if torch.cuda.is_available():
            # Apply real restoration
            return face_restoration.run_codeformer(img, fidelity=0.5)
        return img
    except Exception as e:
        print(f"⚠️ CodeFormer recovery failed: {e}")
        return img


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Skaitch — Forensic Composite Suite</h1>
        <p>Professional generative facial sketching from structured categorical descriptors</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Skaitch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">SDXL · GPU Accelerated</div>',
        unsafe_allow_html=True,
    )

    # ── GPU Status ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        st.info(f"🖥️ **GPU:** {gpu_name}\n\n**VRAM:** {(total-free)/1024**3:.1f}GB / {total/1024**3:.1f}GB")
    else:
        st.warning("⚠️ Running on CPU (Slow)")

    # ── Admin Login ────────────────────────────────────────────────────────
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False

    admin_password_env = os.environ.get("ADMIN_PASSWORD", "")
    
    with st.sidebar.expander("⚙️ Admin Settings", expanded=False):
        if not st.session_state.admin_mode:
            st.markdown("<small>Enter admin password to unlock.</small>", unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password", key="admin_pwd_input", label_visibility="collapsed")
            if st.button("Login", key="admin_login_btn"):
                if pwd and pwd == admin_password_env:
                    st.session_state.admin_mode = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        else:
            st.success("✅ Admin Mode Active")
            # Example Admin specific options
            st.markdown("**Admin Controls**")
            admin_debug = st.checkbox("Show Debug Metrics", value=False)
            admin_save_disabled = st.checkbox("Disable Auto-Save to Disk", value=False)
            
            st.divider()
            st.markdown("**Image Gallery**")
            st.markdown("<small>Recent images saved to `data/`</small>", unsafe_allow_html=True)
            
            if os.path.exists("data"):
                # Get all PNG files, sort by modified time descending
                img_files = [f for f in os.listdir("data") if f.endswith(".png")]
                img_files.sort(key=lambda x: os.path.getmtime(os.path.join("data", x)), reverse=True)
                
                if img_files:
                    # Show up to 10 recent images in the sidebar
                    for img_file in img_files[:10]:
                        with st.container():
                            st.caption(img_file)
                            st.image(os.path.join("data", img_file), use_container_width=True)
                else:
                    st.info("No images found in data folder.")
            else:
                st.info("Data folder does not exist yet.")
                
            st.divider()
            
            if st.button("Logout", key="admin_logout_btn"):
                st.session_state.admin_mode = False
                st.rerun()

    
    # ── Section: Parameters ────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<span class="icon">🎛️</span>'
        '<span class="label">Parameters</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    default_steps = 40
    default_cfg = 12.0

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
        width = st.selectbox("Width", [512, 768, 1024], index=2)
    with col_h:
        height = st.selectbox("Height", [512, 768, 1024], index=2)

    seed = st.number_input(
        "Seed (0 = random)",
        min_value=0,
        max_value=2**32 - 1,
        value=0,
        help="Set a fixed seed to reproduce the same result.",
    )

    # ── Generate ───────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    generate = st.button("🚀  Generate", use_container_width=True, type="primary")

# ─── Main Canvas Features ──────────────────────────────────────────────────────
selected_features: dict[str, str] = {}

# 1. Core Identity & Structure
with st.container(border=True):
    st.markdown('<div class="feature-card-title">👤 Face Structure & Identity</div>', unsafe_allow_html=True)
    col_g, col_a, col_eth, col_st = st.columns(4)
    with col_g: selected_features["Gender"] = st.selectbox("Gender", FACIAL_FEATURES["Gender"])
    with col_a: selected_features["Age range"] = st.selectbox("Age range", FACIAL_FEATURES["Age range"], index=1)
    with col_eth: selected_features["Ethnicity"] = st.selectbox("Ethnicity", FACIAL_FEATURES["Ethnicity"], index=0)
    with col_st: selected_features["Skin tone"] = st.selectbox("Skin tone", FACIAL_FEATURES["Skin tone"], index=2)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_fs, col_jl = st.columns(2)
    with col_fs:
        feature_val = st.selectbox("Face shape", FACIAL_FEATURES["Face shape"])
        selected_features["Face shape"] = feature_val
        st.markdown(get_svg_html(VISUAL_AIDS["Face shape"][feature_val]), unsafe_allow_html=True)
    with col_jl:
        feature_val = st.selectbox("Jawline", FACIAL_FEATURES["Jawline"])
        selected_features["Jawline"] = feature_val
        st.markdown(get_svg_html(VISUAL_AIDS["Jawline"][feature_val]), unsafe_allow_html=True)

# 2. Eyes & Brows
with st.container(border=True):
    st.markdown('<div class="feature-card-title">👁️ Eyes & Brows</div>', unsafe_allow_html=True)
    col_e, col_eb, col_eyebrows = st.columns(3)
    with col_e:
        feature_val = st.selectbox("Eyes shape", FACIAL_FEATURES["Eyes"])
        selected_features["Eyes"] = feature_val
        st.markdown(get_svg_html(VISUAL_AIDS["Eyes"][feature_val]), unsafe_allow_html=True)
    with col_eb:
        selected_features["Eye color"] = st.selectbox("Eye color", FACIAL_FEATURES["Eye color"], index=0)
    with col_eyebrows:
        selected_features["Eyebrows"] = st.selectbox("Eyebrows", FACIAL_FEATURES["Eyebrows"])

# 3. Nose & Mouth
with st.container(border=True):
    st.markdown('<div class="feature-card-title">👃 Nose & Mouth</div>', unsafe_allow_html=True)
    col_n, col_m = st.columns(2)
    with col_n:
        feature_val = st.selectbox("Nose", FACIAL_FEATURES["Nose"])
        selected_features["Nose"] = feature_val
        st.markdown(get_svg_html(VISUAL_AIDS["Nose"][feature_val]), unsafe_allow_html=True)
    with col_m:
        selected_features["Mouth / Lips"] = st.selectbox("Mouth / Lips", FACIAL_FEATURES["Mouth / Lips"])

# 4. Hair & Accessories
with st.container(border=True):
    st.markdown('<div class="feature-card-title">💇 Hair & Accessories</div>', unsafe_allow_html=True)
    col_hs, col_hc, col_fh = st.columns(3)
    with col_hs: selected_features["Hair style"] = st.selectbox("Hair style", FACIAL_FEATURES["Hair style"])
    with col_hc: selected_features["Hair color"] = st.selectbox("Hair color", FACIAL_FEATURES["Hair color"])
    with col_fh: selected_features["Facial hair"] = st.selectbox("Facial hair", FACIAL_FEATURES["Facial hair"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_sp, col_ti = st.columns(2)
    with col_sp: selected_features["Spectacles"] = st.selectbox("Spectacles shape", FACIAL_FEATURES["Spectacles"])
    with col_ti: selected_features["Spectacles Tint"] = st.selectbox("Spectacles tint", FACIAL_FEATURES["Spectacles Tint"])

# 5. Distinguishing Marks & Style
with st.container(border=True):
    st.markdown('<div class="feature-card-title">🔖 Identity Marks & Sketch Style</div>', unsafe_allow_html=True)
    selected_features["Distinguishing marks"] = st.selectbox(
        "Distinguishing marks",
        FACIAL_FEATURES["Distinguishing marks"],
        label_visibility="collapsed",
    )
    col_ss, col_ed = st.columns([1, 2])
    with col_ss:
        sketch_style = st.selectbox("Sketch style", SKETCH_STYLES)
    with col_ed:
        extra_details = st.text_input(
            "Additional details",
            placeholder="e.g. wearing glasses, prominent ears …",
            help="Free-text appended to the generated prompt."
        )
    
    # Build prompt
    prompt, negative_prompt = build_sdxl_forensic_prompt(
        selected_features, sketch_style, extra_details
    )

    # Show prompt preview
    st.markdown('<div class="prompt-preview-label">Generated Prompt Preview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prompt-preview">{prompt}</div>', unsafe_allow_html=True)

# ─── V2 Session State Initialization ──────────────────────────────────────────
if "v2_stage" not in st.session_state:
    st.session_state.v2_stage = "idle"  # idle | drafting | editing | rendering
if "v2_drafts" not in st.session_state:
    st.session_state.v2_drafts = []
if "v2_draft_seeds" not in st.session_state:
    st.session_state.v2_draft_seeds = []
if "v2_selected_sketch" not in st.session_state:
    st.session_state.v2_selected_sketch = None
if "v2_selected_seed" not in st.session_state:
    st.session_state.v2_selected_seed = None
if "v2_edit_history" not in st.session_state:
    st.session_state.v2_edit_history = []
if "v2_features_snapshot" not in st.session_state:
    st.session_state.v2_features_snapshot = {}
if "v2_extra_details" not in st.session_state:
    st.session_state.v2_extra_details = ""
if "v2_sketch_style" not in st.session_state:
    st.session_state.v2_sketch_style = "Pencil sketch"

# ─── Generation logic (STATE 1: DRAFTING) ─────────────────────────────────────
if generate:
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating.")
        st.stop()

    pipe = load_pipeline()

    # Determine seeds and variations
    import random
    num_variations = 3
    base_seed = int(seed) if seed != 0 else random.randint(1, 2**32 - 10)
    seeds_to_run = [base_seed, base_seed + 1, base_seed + 2]

    generated_images = []
    
    with st.spinner("🖌️ Generating with SDXL on T4 GPU ..."):
        for current_seed in seeds_to_run:
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            result = pipe(
                prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pil"
            )
            generated_images.append(result.images[0])

    processed_images = []
    with st.spinner("✨ Running CodeFormer face restoration..."):
        for img in generated_images:
            processed_images.append(run_codeformer(img))

    # Store drafts in session state for variant selection
    st.session_state.v2_stage = "drafting"
    st.session_state.v2_drafts = processed_images
    st.session_state.v2_draft_seeds = seeds_to_run
    st.session_state.v2_features_snapshot = dict(selected_features)
    st.session_state.v2_extra_details = extra_details
    st.session_state.v2_sketch_style = sketch_style
    st.session_state.v2_edit_history = []
    st.session_state.v2_selected_sketch = None

# ─── STATE 1: DRAFTING — Display variants with selection buttons ──────────────
if st.session_state.v2_stage == "drafting" and st.session_state.v2_drafts:
    st.markdown(
        '<div class="success-banner">'
        '<div class="dot"></div>'
        "<span>3 Sketch Variants Generated — Select one to refine</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🖌️ Phase I: Sketch Variants (Select one to edit)")
    
    for idx, img in enumerate(st.session_state.v2_drafts):
        st.markdown('<div class="variant-card">', unsafe_allow_html=True)
        col_img, col_actions = st.columns([2, 1], gap="large")
        
        with col_img:
            st.image(img, use_container_width=True)
            
        with col_actions:
            st.markdown(f"<h3 style='margin-bottom:0.2rem;color:#e2e8f0;'>Variation {idx+1}</h3>", unsafe_allow_html=True)
            st.markdown("<span style='color:#64748b;font-family:\"JetBrains Mono\", monospace;font-size:0.85rem;'>Seed: <span style='color:#06b6d4;'>{}</span></span>".format(st.session_state.v2_draft_seeds[idx]), unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button(f"✅ Select Variation {idx+1}", key=f"select_v_{idx}", use_container_width=True, type="primary"):
                st.session_state.v2_stage = "editing"
                st.session_state.v2_selected_sketch = img
                st.session_state.v2_selected_seed = st.session_state.v2_draft_seeds[idx]
                st.session_state.v2_edit_history = [img]  # Initial history
                st.rerun()
            
            # Inline Download Button
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Variant",
                data=buf.getvalue(),
                file_name=f"skaitch_variant_{idx+1}_{st.session_state.v2_draft_seeds[idx]}.png",
                mime="image/png",
                key=f"dl_v_{idx}",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ─── STATE 2: EDITING — Iterative sketch refinement loop ─────────────────────
elif st.session_state.v2_stage == "editing" and st.session_state.v2_selected_sketch is not None:
    st.markdown(
        '<div class="success-banner">'
        '<div class="dot"></div>'
        f"<span>Editing Mode — Seed {st.session_state.v2_selected_seed}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### ✏️ Phase I: Iterative Sketch Refinement")

    # Show current sketch with Inpainting Canvas overlay
    from streamlit_drawable_canvas import st_canvas
    import numpy as np
    from PIL import Image

    col_sketch, col_controls = st.columns([3, 2], gap="large")
    
    with col_controls:
        st.markdown(
            '<div class="param-card">'
            "<h4>Edit Controls</h4>"
            "</div>",
            unsafe_allow_html=True,
        )
        
        # Brush size controls the mask stroke width
        brush_size = st.slider("Brush Size", 10, 150, 40, key="brush_slider")
        
        edit_instruction = st.text_input(
            "What would you like to change?",
            placeholder="e.g. make the nose more pointed …",
            help="Describe the structural edit. Draw a mask specifically over this area to the left.",
            key="edit_input",
        )

        edit_strength = st.slider(
            "Inpaint Denoise Strength",
            min_value=0.50, # Inpainting requires high noise to reshape geometry
            max_value=1.00,
            value=0.85,
            step=0.05,
            help="At 0.85, SDXL has enough freedom to restructure the masked area entirely.",
            key="edit_strength",
        )

        col_apply, col_undo = st.columns(2)
        with col_apply:
            apply_edit = st.button("🖌️ Apply Edit", use_container_width=True, key="apply_edit")
        with col_undo:
            can_undo = len(st.session_state.v2_edit_history) > 1
            undo_edit = st.button("↩️ Undo", use_container_width=True, key="undo_edit", disabled=not can_undo)

        st.divider()

        col_finalize, col_back = st.columns(2)
        with col_finalize:
            finalize = st.button("🎯 Finalize → Photo", use_container_width=True, key="finalize_sketch", type="primary")
        with col_back:
            go_back = st.button("← Back to Drafts", use_container_width=True, key="back_to_drafts")

    with col_sketch:
        st.markdown("<h4 style='color:#e2e8f0;font-weight:600;'>🖌️ Paint the area to edit</h4>", unsafe_allow_html=True)
        
        bg_image = st.session_state.v2_selected_sketch
        
        # We need a stable display width so the canvas fits in the column,
        # but the actual image behind it is 1024x1024.
        display_width = 800
        aspect_ratio = bg_image.height / bg_image.width
        display_height = int(display_width * aspect_ratio)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",  # Pure white filling
            stroke_width=brush_size,
            stroke_color="#FFFFFF",
            background_image=bg_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="freedraw",
            key="mask_canvas",
        )
        st.caption(f"*Masking Canvas  ·  {len(st.session_state.v2_edit_history)} version(s)*")


    # Handle edit apply
    if apply_edit and edit_instruction.strip():
        # Extact Mask
        mask_pil = None
        has_mask = False
        
        if canvas_result.image_data is not None:
            # image_data is numpy array (H, W, 4). Alpha channel is at index 3.
            mask_array = canvas_result.image_data[:, :, 3]
            # Check if any pixels were actually drawn
            if np.any(mask_array > 0):
                has_mask = True
                # Convert to strict 0/255 mask.
                mask_binary = (mask_array > 0).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_binary).convert("L")
                # Resize mask to original sketch resolution (e.g., 1024x1024)
                mask_pil = mask_pil.resize(st.session_state.v2_selected_sketch.size, Image.Resampling.LANCZOS)
        
        if not has_mask:
            st.error("⚠️ Please draw a mask on the image to specify where the edit should occur.")
        else:
            pipe = load_pipeline()
            from prompt_builder import build_edit_prompt
            from sketch_refiner import run_sketch_edit
            
            edit_prompt, edit_neg = build_edit_prompt(
                st.session_state.v2_features_snapshot,
                edit_instruction,
                st.session_state.v2_sketch_style,
            )
            
            with st.spinner(f"🖌️ Applying masked inpaint edit (Strength: {edit_strength}) …"):
                edited_sketch = run_sketch_edit(
                    pipe=pipe,
                    sketch_pil=st.session_state.v2_selected_sketch,
                    mask_pil=mask_pil,
                    edit_prompt=edit_prompt,
                    negative_prompt=edit_neg,
                    strength=edit_strength,
                )
                
                # Fix 4: Force UI Cache Busting
                edited_sketch = edited_sketch.copy()
            
            st.session_state.v2_edit_history.append(edited_sketch)
            st.session_state.v2_selected_sketch = edited_sketch
            st.rerun()

    # Handle undo
    if undo_edit and can_undo:
        st.session_state.v2_edit_history.pop()
        st.session_state.v2_selected_sketch = st.session_state.v2_edit_history[-1]
        st.rerun()

    # Handle back to drafts
    if go_back:
        st.session_state.v2_stage = "drafting"
        st.session_state.v2_selected_sketch = None
        st.session_state.v2_edit_history = []
        st.rerun()

    # Handle finalize → trigger rendering
    if finalize:
        st.session_state.v2_stage = "rendering"
        st.rerun()

# ─── STATE 3: RENDERING — Photorealistic ControlNet pass ─────────────────────
elif st.session_state.v2_stage == "rendering" and st.session_state.v2_selected_sketch is not None:
    st.markdown(
        '<div class="success-banner">'
        '<div class="dot"></div>'
        "<span>Rendering Photorealistic Output</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    main_image = st.session_state.v2_selected_sketch
    features_snap = st.session_state.v2_features_snapshot
    extra_snap = st.session_state.v2_extra_details

    col_sketch_final, col_photo = st.columns([2, 3], gap="large")

    with col_sketch_final:
        st.markdown("<h4 style='color:#94a3b8;font-weight:600;'>🖌️ Finalized Sketch</h4>", unsafe_allow_html=True)
        st.image(main_image, use_container_width=True)
        st.caption(f"*Seed {st.session_state.v2_selected_seed}  ·  {len(st.session_state.v2_edit_history)} edit(s)*")

    with col_photo:
        st.markdown("<h4 style='color:#06b6d4;font-weight:700;'>📸 Photorealistic Phase II Refinement</h4>", unsafe_allow_html=True)
        with st.spinner("🤖 SDXL-ControlNet Refinement — Generating Photorealistic Output …"):
            # Clear VRAM before Phase II to ensure maximum headroom
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                refinement_image = refinement_pipeline.run_sdxl_refinement(main_image, features_snap, extra_snap)
                refine_success = True
            except Exception as e:
                import html as html_mod
                refine_error = html_mod.escape(str(e))
                refine_success = False

        if refine_success:
            st.markdown('<div class="featured-output">', unsafe_allow_html=True)
            st.image(refinement_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("*✨ SDXL-ControlNet High-Resolution Refinement ✨*")
        else:
            st.error(f"Refinement failed: {refine_error}\n\nEnsure that the ControlNet weights are correctly downloaded to the NVMe storage.")

    # ── Auto-Save to data/ ──────────────────────────────────────────────
    import datetime
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    admin_save_disabled = st.session_state.get("admin_mode", False) and st.session_state.get("Disable Auto-Save to Disk", False)
    
    if not admin_save_disabled:
        sketch_path = os.path.join("data", f"sketch_{timestamp}_final.png")
        main_image.save(sketch_path)
        
        if refine_success:
            refine_save_path = os.path.join("data", f"refinement_{timestamp}.png")
            refinement_image.save(refine_save_path)

    # ── Meta info & Downloads ──────────────────────────────────────────
    st.divider()
    col_meta, col_dl = st.columns([2, 3], gap="large")
    
    with col_meta:
        st.markdown(
            '<div class="param-card">'
            "<h4>Parameters</h4>"
            '<div class="param-row">'
            '<span class="key">Mode</span>'
            '<span class="val">Forensic V2</span>'
            "</div>"
            '<div class="param-row">'
            '<span class="key">Edits Applied</span>'
            f'<span class="val">{len(st.session_state.v2_edit_history) - 1}</span>'
            "</div>"
            '<div class="param-row">'
            '<span class="key">Seed</span>'
            f'<span class="val">{st.session_state.v2_selected_seed}</span>'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_dl:
        buf_sketch = io.BytesIO()
        main_image.save(buf_sketch, format="PNG")
        st.download_button(
            label="⬇️ Download Finalized Sketch (PNG)",
            data=buf_sketch.getvalue(),
            file_name=f"skaitch_sketch_{timestamp}_final.png",
            mime="image/png",
            use_container_width=True,
            key="dl_final_sketch"
        )
        
        if refine_success:
            st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
            buf_refine = io.BytesIO()
            refinement_image.save(buf_refine, format="PNG")
            st.download_button(
                label="⬇️ Download Refinement (PNG)",
                data=buf_refine.getvalue(),
                file_name=f"skaitch_refinement_{timestamp}.png",
                mime="image/png",
                use_container_width=True,
                key="dl_refinement"
            )

    # Option to start over
    st.divider()
    if st.button("🔄 Start New Composite", use_container_width=True, key="reset_v2"):
        st.session_state.v2_stage = "idle"
        st.session_state.v2_drafts = []
        st.session_state.v2_selected_sketch = None
        st.session_state.v2_edit_history = []
        st.rerun()

# ─── IDLE state ───────────────────────────────────────────────────────────────
elif st.session_state.v2_stage == "idle":
    gpu_txt = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.info(f"**Ready!** System is loaded with `SDXL Base 1.0`, `SDXL ControlNet`, and `CodeFormer` on **{gpu_txt}**.")

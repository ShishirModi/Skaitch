# File: app.py
# Purpose: Streamlit frontend for Stable Diffusion image generation

import streamlit as st
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
    /* ── Typography ──────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Restore Streamlit icon fonts to prevent ligatures rendering as text */
    [class*="material-symbols"],
    [data-testid*="Icon"],
    [data-testid*="stIcon"],
    .st-icon {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }

    /* ── Global tweaks ───────────────────────────────────────────────── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: clamp(1.5rem, 5vw, 2.4rem) clamp(1.5rem, 5vw, 2.8rem);
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
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.6;
        font-size: clamp(0.85rem, 2.5vw, 0.95rem);
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1f 0%, #161636 100%) !important;
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

    forensic_mode = True
    
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
    # Ethnicity + Skin Tone row
    col_eth, col_st = st.columns(2)
    with col_eth:
        selected_features["Ethnicity"] = st.selectbox(
            "Ethnicity", FACIAL_FEATURES["Ethnicity"], index=0
        )
    with col_st:
        selected_features["Skin tone"] = st.selectbox(
            "Skin tone", FACIAL_FEATURES["Skin tone"], index=2
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
        feature_val = st.selectbox("Eyes shape", FACIAL_FEATURES["Eyes"])
        selected_features["Eyes"] = feature_val
        st.markdown(get_svg_html(VISUAL_AIDS["Eyes"][feature_val]), unsafe_allow_html=True)

    with col_eb:
        selected_features["Eye color"] = st.selectbox(
            "Eye color", FACIAL_FEATURES["Eye color"], index=0
        )

    # Eyebrows row
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

    # (Skin tone removed from here, moved up to Ethnicity row)

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

    # ── Section: Accessories ──────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section">'
        '<span class="icon">👓</span>'
        '<span class="label">Accessories</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_sp, col_ti = st.columns(2)
    with col_sp:
        selected_features["Spectacles"] = st.selectbox(
            "Spectacles shape", FACIAL_FEATURES["Spectacles"]
        )
    with col_ti:
        selected_features["Spectacles Tint"] = st.selectbox(
            "Spectacles tint", FACIAL_FEATURES["Spectacles Tint"]
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
    prompt, negative_prompt = build_sdxl_forensic_prompt(
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
    default_steps = 40
    default_cfg = 12.0

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
    generate = st.button("🚀  Generate", use_container_width=True)

# ─── Generation logic ─────────────────────────────────────────────────────────
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

    # The primary image for downstream tasks
    main_image = processed_images[0]

    # ── Success banner ─────────────────────────────────────────────────
    st.markdown(
        '<div class="success-banner">'
        '<div class="dot"></div>'
        f"<span>{num_variations} Image{'s' if num_variations>1 else ''} generated safely</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Pipeline specific displays ──────────────────────────────────────
    st.markdown("#### Stable Diffusion Sketches (SDXL + CodeFormer)")
    cols = st.columns(3, gap="medium")
    for idx, (img, col) in enumerate(zip(processed_images, cols)):
        with col:
            st.image(img, use_container_width=True)
            st.caption(f"*Variation {idx+1}*")
    
    st.divider()
    col_img_refine, col_meta = st.columns([3, 2], gap="large")

    with col_img_refine:
        st.markdown("#### Photorealistic Refinement (Variation 1)")
        with st.spinner("🤖 SDXL-ControlNet Refinement — Generating Photorealistic Output …"):
            # Clear VRAM before Phase II to ensure maximum headroom
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                refinement_image = refinement_pipeline.run_sdxl_refinement(main_image, selected_features, extra_details)
                refine_success = True
            except Exception as e:
                import html
                refine_error = html.escape(str(e))
                refine_success = False

        if refine_success:
            st.image(refinement_image, use_container_width=True)
            st.caption(f"*SDXL-ControlNet Refinement*")
        else:
            st.error(f"Refinement failed: {refine_error}\n\nEnsure that the ControlNet weights are correctly downloaded to the NVMe storage.")

    # ── Auto-Save to data/ ──────────────────────────────────────────────
    import datetime
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check admin control before saving 
    admin_save_disabled = st.session_state.get("admin_mode", False) and st.session_state.get("Disable Auto-Save to Disk", False)
    
    if not admin_save_disabled:
        for idx, img in enumerate(processed_images):
            sketch_path = os.path.join("data", f"sketch_{timestamp}_v{idx+1}.png")
            img.save(sketch_path)
        
        # Save the Refinement output if applicable
        if refine_success:
            refine_save_path = os.path.join("data", f"refinement_{timestamp}.png")
            refinement_image.save(refine_save_path)

    # ── Meta info (Right Column) ──────────────────────────────────────────
    with col_meta:
        seed_row = ""
        if seed != 0:
            seed_row = (
                '<div class="param-row">'
                '<span class="key">Base Seed</span>'
                f'<span class="val">{base_seed}</span>'
                "</div>"
            )
        mode_row = ""
        mode_row = (
            '<div class="param-row">'
            '<span class="key">Mode</span>'
            '<span class="val">Forensic (3x)</span>'
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

        for idx, img in enumerate(processed_images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label=f"⬇️ Download Sketch {idx+1} (PNG)",
                data=buf.getvalue(),
                file_name=f"skaitch_sketch_{timestamp}_v{idx+1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_sketch_{idx}"
            )
        
        if refine_success:
            st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
            buf2 = io.BytesIO()
            refinement_image.save(buf2, format="PNG")
            st.download_button(
                label="⬇️ Download Refinement (PNG)",
                data=buf2.getvalue(),
                file_name=f"skaitch_refinement_{timestamp}.png",
                mime="image/png",
                use_container_width=True,
                key="dl_refinement"
            )

else:
    gpu_txt = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    st.info(f"**Ready!** System is loaded with `SDXL Base 1.0`, `SDXL ControlNet`, and `CodeFormer` on **{gpu_txt}**.")

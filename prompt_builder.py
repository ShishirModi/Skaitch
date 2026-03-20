# File: prompt_builder.py
# Purpose: Forensic sketch prompt builder for Stable Diffusion

# ─── Facial Feature Options ───────────────────────────────────────────────────
# Each key maps to a list of selectable options displayed in the UI.

FACIAL_FEATURES = {
    "Gender": ["Male", "Female"],
    "Age range": ["18–25", "26–35", "36–45", "46–55", "56–65", "65+"],
    "Face shape": [
        "Oval", "Round", "Square", "Heart", "Oblong", "Diamond", "Triangle", "Pear"
    ],
    "Eyes": [
        "Almond", "Round", "Hooded", "Deep-set", "Monolid",
        "Wide-set", "Close-set", "Upturned", "Downturned"
    ],
    "Eyebrows": [
        "Thick", "Thin", "Arched", "Straight", "Bushy", "Sparse",
        "High arch", "Soft arch", "S-shaped"
    ],
    "Nose": [
        "Straight", "Broad", "Narrow", "Aquiline", "Button",
        "Wide bridge", "Snub", "Roman", "Bulbous", "Hawk"
    ],
    "Mouth / Lips": [
        "Thin", "Full", "Wide", "Small", "Downturned", "Upturned",
        "Bow-shaped", "Heavy lower lip", "Heavy upper lip"
    ],
    "Jawline": [
        "Strong", "Soft", "Pointed", "Wide", "Receding", "V-shaped"
    ],
    "Hair style": [
        "Short cropped", "Buzz cut", "Slicked back", "Curly", "Wavy",
        "Straight long", "Bald", "Receding hairline", "Ponytail",
        "Braids", "Afro", "Undercut",
    ],
    "Hair color": [
        "Black", "Dark brown", "Light brown", "Auburn", "Blonde",
        "Grey", "White", "Red",
    ],
    "Facial hair": [
        "None", "Stubble", "Full beard", "Goatee", "Mustache",
        "Van Dyke", "Soul patch",
    ],
    "Ethnicity": [
        "Caucasian / White", 
        "African / Black", 
        "East Asian", 
        "South Asian", 
        "Southeast Asian", 
        "Middle Eastern / North African", 
        "Hispanic / Latino", 
        "Native American / Indigenous",
        "Pacific Islander",
        "Mixed / Multiracial"
    ],
    "Skin tone": [
        "Fair", "Light", "Medium", "Olive", "Tan", "Brown", "Dark",
    ],
    "Eye color": [
        "Brown", "Blue", "Green", "Hazel", "Amber", "Grey", "Black"
    ],
    "Spectacles": [
        "None", "Rectangular", "Round", "Oval", "Square", "Aviator", "Cat-eye"
    ],
    "Spectacles Tint": [
        "None", "Transparent", "Lightly Tinted", "Dark Sunglasses"
    ],
    "Distinguishing marks": [
        "None",
        "Scar on left cheek",
        "Scar on right cheek",
        "Scar on forehead",
        "Scar across nose",
        "Mole near mouth",
        "Mole near eye",
        "Birthmark on cheek",
        "Acne scars",
        "Freckles",
    ],
}

# ─── Sketch Style Presets ─────────────────────────────────────────────────────

SKETCH_STYLES = [
    "Pencil sketch",
    "Charcoal sketch",
    "Police composite",
    "Forensic artist rendering",
]

# ─── Negative Prompt ──────────────────────────────────────────────────────────

FORENSIC_NEGATIVE = (
    "color photograph, oil painting, cartoon, anime, 3d render, CGI, watercolor, "
    "blurry, low quality, low resolution, pixelated, distorted face, asymmetric face, "
    "extra fingers, deformed features, watermark, signature, text, frame, border, "
    "background clutter, multiple people, accessories, jewelry, glasses"
)

# ─── Recommended Defaults ────────────────────────────────────────────────────

FORENSIC_DEFAULTS = {
    "guidance_scale": 10.0,
    "num_inference_steps": 30,
}

# ─── Prompt Builder ──────────────────────────────────────────────────────────


def build_forensic_prompt(
    features: dict[str, str],
    style: str = "Pencil sketch",
    extra_details: str = "",
) -> tuple[str, str]:
    """Assemble a forensic-quality SD prompt from individual facial features.

    Args:
        features: dict mapping category name → selected option
                  (keys should match FACIAL_FEATURES keys).
        style: one of SKETCH_STYLES.
        extra_details: optional free-text appended to the prompt.

    Returns:
        (prompt, negative_prompt) tuple ready for the SD pipeline.
    """

    gender = features.get("Gender", "Male")
    age = features.get("Age range", "26–35")

    # Opening phrase
    parts = [
        f"{style} of a {gender.lower()} face",
        f"age {age}",
    ]

    # Face structure
    face_shape = features.get("Face shape")
    if face_shape:
        parts.append(f"{face_shape.lower()} face shape")

    jawline = features.get("Jawline")
    if jawline:
        parts.append(f"{jawline.lower()} jawline")

    # Skin and Ethnicity
    ethnicity = features.get("Ethnicity")
    skin = features.get("Skin tone")
    if ethnicity:
        parts.append(f"{ethnicity.lower()} ethnicity")
    if skin:
        parts.append(f"{skin.lower()} skin tone")

    # Eyes & brows
    eyes = features.get("Eyes")
    eye_color = features.get("Eye color")
    if eye_color:
        parts.append(f"{eye_color.lower()} eyes")
    if eyes:
        parts.append(f"{eyes.lower()} shaped eyes")

    eyebrows = features.get("Eyebrows")
    if eyebrows:
        parts.append(f"{eyebrows.lower()} eyebrows")

    # Nose & Mouth
    nose = features.get("Nose")
    if nose:
        parts.append(f"{nose.lower()} nose")

    mouth = features.get("Mouth / Lips")
    if mouth:
        parts.append(f"{mouth.lower()} lips")

    # Hair
    hair_style = features.get("Hair style")
    hair_color = features.get("Hair color")
    if hair_style and hair_style != "Bald":
        hair_desc = f"{hair_color.lower()} {hair_style.lower()} hair" if hair_color else f"{hair_style.lower()} hair"
        parts.append(hair_desc)
    elif hair_style == "Bald":
        parts.append("bald head")

    # Facial hair
    facial_hair = features.get("Facial hair")
    if facial_hair and facial_hair != "None":
        parts.append(facial_hair.lower())

    # Distinguishing marks
    marks = features.get("Distinguishing marks")
    if marks and marks != "None":
        parts.append(marks.lower())

    # Spectacles
    glasses = features.get("Spectacles")
    tint = features.get("Spectacles Tint")
    if glasses and glasses != "None":
        tint_str = f" with {tint.lower()} lenses" if tint and tint != "None" else ""
        parts.append(f"wearing {glasses.lower()} shaped spectacles{tint_str}")

    # Style tokens for quality
    quality_tokens = (
        "front-facing portrait, centered composition, white background, "
        "ultra-high detail, sharp facial lines, professional forensic composite, "
        "photorealistic pencil rendering, 8k resolution, studio lighting, "
        "criminal investigation, legal evidence accuracy"
    )

    prompt = ", ".join(parts) + ", " + quality_tokens

    # Extra details from user
    if extra_details.strip():
        prompt += ", " + extra_details.strip()

    return prompt, FORENSIC_NEGATIVE

def build_sdxl_forensic_prompt(
    features: dict[str, str],
    style: str = "Pencil sketch",
    extra_details: str = "",
) -> tuple[str, str]:
    """Wrapper that calls build_forensic_prompt and appends SDXL boosters."""
    prompt, neg = build_forensic_prompt(features, style, extra_details)
    return prompt + ", best quality, masterpiece, highly detailed face", neg

def build_refinement_prompt(
    features: dict[str, str],
    extra_details: str = "",
) -> tuple[str, str]:
    """Assemble a high-fidelity photorealistic prompt for Phase II refinement.
    
    Reuses the same feature logic as build_forensic_prompt but swaps
    sketch-specific tokens for photographic ones.
    """
    gender = features.get("Gender", "person")
    age = features.get("Age range", "adult")

    parts = [
        f"Extreme close-up professional studio portrait of a {age} {gender.lower()}",
    ]

    # Re-use most of the descriptive parts
    for cat in ["Ethnicity", "Skin tone", "Face shape", "Jawline", "Eyes", "Eyebrows", "Nose", "Mouth / Lips"]:
        val = features.get(cat)
        if val:
            if cat == "Mouth / Lips":
                parts.append(f"{val.lower()} lips")
            elif cat == "Ethnicity":
                parts.append(f"{val.lower()} ethnicity")
            elif cat == "Eyes":
                parts.append(f"{val.lower()} shaped eyes")
            else:
                parts.append(f"{val.lower()} {cat.lower()}")

    # Chromatic Details (Weighted for Phase II)
    eye_color = features.get("Eye color")
    if eye_color:
        parts.append(f"({eye_color.lower()} eyes:1.6)")

    # Hair (Weighted for better adherence in Refinement)
    hair_style = features.get("Hair style")
    hair_color = features.get("Hair color")
    if hair_style and hair_style != "Bald":
        hair_desc = f"({hair_color.lower()} {hair_style.lower()} hair:1.3)" if hair_color else f"({hair_style.lower()} hair:1.3)"
        parts.append(hair_desc)
    elif hair_style == "Bald":
        parts.append("(bald head:1.3)")

    # Facial hair
    facial_hair = features.get("Facial hair")
    if facial_hair and facial_hair != "None":
        parts.append(facial_hair.lower())

    # Distinguishing marks
    marks = features.get("Distinguishing marks")
    if marks and marks != "None":
        parts.append(marks.lower())

    # Spectacles (Weighted for Phase II)
    glasses = features.get("Spectacles")
    tint = features.get("Spectacles Tint")
    if glasses and glasses != "None":
        tint_str = f" with {tint.lower()} lenses" if tint and tint != "None" else ""
        parts.append(f"(wearing {glasses.lower()} shaped spectacles{tint_str}:1.3)")

    # Quality tokens for photorealism
    photo_tokens = (
        "hyper-realistic face, highly detailed skin texture, pores, 8k resolution, "
        "cinematic lighting, masterpiece, sharp focus, professional photography, "
        "detailed eyes, detailed hair texture"
    )
    if hair_color:
        photo_tokens += f", {hair_color.lower()} hair color"
    if eye_color:
        photo_tokens += f", {eye_color.lower()} eye color"

    prompt = ", ".join(parts) + ", " + photo_tokens

    if extra_details.strip():
        # Strip potential "pencil" or "sketch" references from extra_details if present
        details = extra_details.strip().lower()
        if "sketch" not in details and "pencil" not in details:
            prompt += ", " + extra_details.strip()

    negative_prompt = (
        "scribble, sketch, drawing, painting, pencil, charcoal, cartoon, anime, 3d, monochromatic, "
        "blurry, low quality, distorted face, watermark, text, signature"
    )

    return prompt, negative_prompt

def build_sdxl_refinement_prompt(
    features: dict[str, str],
    extra_details: str = "",
) -> tuple[str, str]:
    """Wrapper for SDXL refinement pass."""
    prompt, neg = build_refinement_prompt(features, extra_details)
    return prompt + ", best quality, highly detailed face", neg

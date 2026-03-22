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
    "background clutter, multiple people, jewelry"
)

# ─── Recommended Defaults ────────────────────────────────────────────────────

FORENSIC_DEFAULTS = {
    "guidance_scale": 10.0,
    "num_inference_steps": 30,
}

# ─── Prompt Builder ──────────────────────────────────────────────────────────


def _build_narrative(features: dict[str, str], is_photo: bool) -> list[str]:
    """Helper to construct the grammatical narrative sentences. Correctly handles pronouns."""
    gender = features.get("Gender", "Male").lower()
    subject = "He" if gender == "male" else ("She" if gender == "female" else "They")
    possessive = "His" if gender == "male" else ("Her" if gender == "female" else "Their")

    age = features.get("Age range", "26–35")
    ethnicity = features.get("Ethnicity", "")
    skin = features.get("Skin tone", "")

    # Sentence 1: The Base Noun Phrase
    eth_str = f" {ethnicity.lower()} " if ethnicity and ethnicity != "None" else " "
    base_noun = f"{eth_str.strip()} {gender}" if eth_str.strip() else f"{gender}"
    sentence1 = f"of a {base_noun}, age {age}."
    
    # Sentence 2: Structure
    face_shape = features.get("Face shape", "")
    jawline = features.get("Jawline", "")
    struct_parts = []
    if skin and skin != "None": struct_parts.append(f"{skin.lower()} skin")
    if face_shape and face_shape != "None": struct_parts.append(f"an {face_shape.lower()} face shape")
    if jawline and jawline != "None": struct_parts.append(f"a {jawline.lower()} jawline")
    
    sentence2 = f"{subject} features " + ", ".join(struct_parts) + "." if struct_parts else ""

    # Sentence 3: Eyes & Nose
    eyes = features.get("Eyes", "")
    eye_color = features.get("Eye color", "")
    eyebrows = features.get("Eyebrows", "")
    nose = features.get("Nose", "")
    mouth = features.get("Mouth / Lips", "")

    if is_photo and eye_color:
        eyes_desc = f"({eye_color.lower()} {eyes.lower()} shaped eyes:1.4)"
    else:
        eyes_desc = f"{eye_color.lower()} {eyes.lower()} shaped eyes" if eye_color else f"{eyes.lower()} shaped eyes"
        
    sentence3 = f"{possessive} facial features include {eyes_desc} beneath {eyebrows.lower()} eyebrows, leading down to a {nose.lower()} nose and {mouth.lower()} lips."

    # Sentence 4: Hair & Dynamic Extras (only appended if required)
    hair_style = features.get("Hair style", "")
    hair_color = features.get("Hair color", "")
    facial_hair = features.get("Facial hair", "")
    marks = features.get("Distinguishing marks", "")
    glasses = features.get("Spectacles", "")
    tint = features.get("Spectacles Tint", "")

    hair_str = f"{hair_color.lower()} {hair_style.lower()} hair" if hair_color else f"{hair_style.lower()} hair"
    if is_photo and hair_style != "Bald":
        hair_str = f"({hair_str}:1.3)"
        
    hair_desc = hair_str if hair_style != "Bald" else "a bald head"
    
    extras = []
    if facial_hair and facial_hair != "None": extras.append(f"a {facial_hair.lower()}")
    if marks and marks != "None": extras.append(f"{marks.lower()}")
    if glasses and glasses != "None":
        tint_str = f" with {tint.lower()} lenses" if tint and tint != "None" else ""
        glass_str = f"wearing {glasses.lower()} shaped spectacles{tint_str}"
        if is_photo: glass_str = f"({glass_str}:1.3)"
        extras.append(glass_str)
        
    sentence4 = f"{subject} has {hair_desc}"
    if extras:
        if len(extras) == 1:
            sentence4 += f", and is also characterized by {extras[0]}."
        else:
            sentence4 += f", and is characterized by {', '.join(extras[:-1])}, and {extras[-1]}."
    else:
        sentence4 += "."

    return [s for s in [sentence1, sentence2, sentence3, sentence4] if s]


def build_forensic_prompt(
    features: dict[str, str],
    style: str = "Pencil sketch",
    extra_details: str = "",
) -> tuple[str, str]:
    """Assemble a forensic-quality SD prompt from individual facial features."""
    
    # Generate grammatical narrative
    sentences = _build_narrative(features, is_photo=False)
    
    # Prefix medium
    prompt = f"A highly detailed {style.lower()} {sentences[0]} {' '.join(sentences[1:])}"

    # Style tokens for quality
    quality_tokens = (
        "front-facing portrait, centered composition, white background, "
        "ultra-high detail, sharp facial lines, professional forensic composite, "
        "photorealistic pencil rendering, 8k resolution, studio lighting, "
        "criminal investigation, legal evidence accuracy"
    )

    prompt = prompt + ", " + quality_tokens

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
    """Assemble a high-fidelity photorealistic prompt for Phase II refinement."""
    
    # Generate grammatical narrative with photorealism weighting
    sentences = _build_narrative(features, is_photo=True)
    
    # Prefix medium
    prompt = f"Extreme close-up professional studio portrait {sentences[0]} {' '.join(sentences[1:])}"

    # Quality tokens for photorealism
    photo_tokens = (
        "hyper-realistic face, highly detailed skin texture, pores, 8k resolution, "
        "cinematic lighting, masterpiece, sharp focus, professional photography, "
        "detailed eyes, detailed hair texture"
    )

    prompt = prompt + ", " + photo_tokens

    if extra_details.strip():
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


def build_edit_prompt(
    features: dict[str, str],
    edit_instruction: str,
    style: str = "Pencil sketch",
) -> tuple[str, str]:
    """Build a prompt for iterative sketch editing (SDXL i2i).

    Combines the original forensic feature prompt with a specific
    edit instruction from the operator. The edit instruction is
    given elevated weighting to ensure it takes effect even at
    low denoise strengths.

    Args:
        features: dict mapping category name → selected option.
        edit_instruction: Free-text edit (e.g., "make the nose more pointed").
        style: The sketch style to maintain.

    Returns:
        (prompt, negative_prompt) tuple for the i2i pass.
    """
    # In V2.1, Regional Inpainting prevents semantic locking, so we can pass the
    # full grammatical narrative to ensure unmasked context is perfectly understood.
    sentences = _build_narrative(features, is_photo=False)
    base_prompt = f"A highly detailed {style.lower()} {sentences[0]} {' '.join(sentences[1:])}"

    # Append the heavily weighted edit instruction
    if edit_instruction.strip():
        base_prompt += f", ({edit_instruction.strip()}:1.5)"
        
    # Append boosters
    base_prompt += ", best quality, masterpiece, highly detailed face"

    return base_prompt, FORENSIC_NEGATIVE

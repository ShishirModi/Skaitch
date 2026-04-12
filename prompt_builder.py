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

# §1.4 fix: Removed "color photograph" from negative prompt. It contradicted the
# positive prompt's "photorealistic pencil rendering" token, creating a semantic
# tug-of-war that caused inconsistent sketch vs. semi-realistic outputs.
FORENSIC_NEGATIVE = (
    "oil painting, cartoon, anime, 3d render, CGI, watercolor, "
    "blurry, low quality, low resolution, pixelated, distorted face, asymmetric face, "
    "extra fingers, deformed features, watermark, signature, text, frame, border, "
    "background clutter, multiple people, jewelry, colored skin, color image"
)

# ─── Recommended Defaults ────────────────────────────────────────────────────

FORENSIC_DEFAULTS = {
    "guidance_scale": 10.0,
    "num_inference_steps": 30,
}

# ─── Improvement 1.5: Ethnicity-Specific Anatomical Grounding ─────────────────
# Each ethnicity maps to anatomically-informed descriptors that improve coherence
ETHNICITY_ANATOMICAL_DESCRIPTORS = {
    "Caucasian / White": [
        "varied nose bridge, prominent cheekbones, diverse eye set",
    ],
    "African / Black": [
        "fuller lips, broader nose structure, prominent nasal base, diverse eye setting, rich skin undertones",
    ],
    "East Asian": [
        "straighter nose bridge, prominent cheekbones, varied eye set including monolid variations",
    ],
    "South Asian": [
        "darker iris, distinctive brow positioning, varied nose structure and bridge, varied eye set",
    ],
    "Southeast Asian": [
        "moderate nose bridge, varied eye set, distinctive cheekbone prominence, diverse undertones",
    ],
    "Middle Eastern / North African": [
        "pronounced nose bridge, defined cheekbones, varied eye set, rich skin undertones",
    ],
    "Hispanic / Latino": [
        "varied nose structure, diverse cheekbone prominence, varied eye set, warm skin undertones",
    ],
    "Native American / Indigenous": [
        "distinctive cheekbone structure, varied nose bridge, varied eye set, warm undertones",
    ],
    "Pacific Islander": [
        "rounded facial features, diverse cheekbones, varied eye set, warm undertones",
    ],
    "Mixed / Multiracial": [
        "diverse facial feature combination, varied eye set, mixed undertones",
    ],
}

# ─── Improvement 2.3: Prompt-Based Mask Region Inference ────────────────────────
# Maps edit verbs/nouns to anatomical regions for auto-mask suggestions.
#
# §2.5 fix: Replaced flat dict (which silently dropped duplicate keys like "wider")
# with a list-of-tuples structure. Each entry is (keyword, mapping). This preserves
# all mappings and allows context-aware disambiguation in infer_mask_region_from_edit().
EDIT_REGION_MAPPING = [
    # Nose edits — high-priority compound phrases listed FIRST so they match before
    # the ambiguous single-word entries.
    ("nose wider", {"region": "nose", "dilate_px": 35, "buffer": "cheeks"}),
    ("nose narrower", {"region": "nose", "dilate_px": 25, "buffer": "cheeks"}),
    ("nose", {"region": "nose", "dilate_px": 25, "buffer": "cheeks_forehead"}),
    ("nostr", {"region": "nose", "dilate_px": 25, "buffer": "cheeks_forehead"}),
    ("bridge", {"region": "nose", "dilate_px": 20, "buffer": "cheeks_forehead"}),
    ("sharper", {"region": "nose", "dilate_px": 30, "buffer": "cheeks_forehead"}),
    ("pointed", {"region": "nose", "dilate_px": 20, "buffer": "cheeks_forehead"}),
    ("aquiline", {"region": "nose", "dilate_px": 25, "buffer": "cheeks_forehead"}),

    # Jawline edits
    ("jawline", {"region": "jaw", "dilate_px": 40, "buffer": "cheeks_neck"}),
    ("jaw", {"region": "jaw", "dilate_px": 40, "buffer": "cheeks_neck"}),
    ("chin", {"region": "jaw", "dilate_px": 35, "buffer": "neck"}),
    ("stronger", {"region": "jaw", "dilate_px": 45, "buffer": "cheeks_neck"}),
    ("weaker", {"region": "jaw", "dilate_px": 35, "buffer": "cheeks_neck"}),

    # Eye edits
    ("eyes wider", {"region": "eyes", "dilate_px": 35, "buffer": "eyebrows_cheeks"}),
    ("eye wider", {"region": "eyes", "dilate_px": 35, "buffer": "eyebrows_cheeks"}),
    ("eyes", {"region": "eyes", "dilate_px": 30, "buffer": "eyebrows_cheeks"}),
    ("eye", {"region": "eyes", "dilate_px": 30, "buffer": "eyebrows_cheeks"}),
    ("iris", {"region": "eyes", "dilate_px": 15, "buffer": "none"}),
    ("larger", {"region": "eyes", "dilate_px": 35, "buffer": "eyebrows_cheeks"}),
    ("smaller", {"region": "eyes", "dilate_px": 25, "buffer": "eyebrows_cheeks"}),

    # Mouth edits
    ("mouth", {"region": "mouth", "dilate_px": 25, "buffer": "chin"}),
    ("lips", {"region": "mouth", "dilate_px": 25, "buffer": "chin"}),
    ("fuller", {"region": "mouth", "dilate_px": 30, "buffer": "chin"}),
    ("thinner", {"region": "mouth", "dilate_px": 20, "buffer": "chin"}),

    # Hair edits
    ("hairstyle", {"region": "hair", "dilate_px": 60, "buffer": "forehead"}),
    ("hair", {"region": "hair", "dilate_px": 60, "buffer": "forehead"}),

    # Cheek edits
    ("cheeks", {"region": "cheeks", "dilate_px": 40, "buffer": "eyes_jaw"}),
    ("cheek", {"region": "cheeks", "dilate_px": 40, "buffer": "eyes_jaw"}),

    # Forehead edits
    ("forehead", {"region": "forehead", "dilate_px": 35, "buffer": "hair"}),
]

# ─── Feature Complexity Weights for Adaptive Guidance (Improvement 1.3) ───────
FEATURE_GUIDANCE_WEIGHTS = {
    "Gender": 1.0,
    "Age range": 1.0,
    "Face shape": 1.1,
    "Eyes": 1.2,           # Complex feature
    "Eyebrows": 1.05,
    "Nose": 1.15,          # Very complex feature
    "Mouth / Lips": 1.1,   # Complex feature
    "Jawline": 1.1,        # Complex feature
    "Hair style": 0.95,    # Simple (high-level)
    "Hair color": 0.9,     # Simple (straightforward)
    "Facial hair": 1.05,
    "Ethnicity": 1.0,
    "Skin tone": 0.9,      # Simple
    "Eye color": 1.05,
    "Spectacles": 1.0,
    "Spectacles Tint": 0.85,
    "Distinguishing marks": 1.1,
}

# ─── Prompt Builder ──────────────────────────────────────────────────────────


def _get_ethnicity_anatomical_boost(ethnicity: str) -> str:
    """Return ethnicity-specific anatomical descriptors (Improvement 1.5)."""
    if ethnicity and ethnicity != "None":
        descriptors = ETHNICITY_ANATOMICAL_DESCRIPTORS.get(ethnicity, [])
        if descriptors:
            return descriptors[0]
    return ""


def compute_adaptive_guidance_scale(features: dict[str, str], base_guidance: float = 12.0) -> float:
    """Compute adaptive guidance scale based on feature complexity (Improvement 1.3).

    Simple features (hair style, color) use lower guidance to avoid over-constraint.
    Complex features (nose, eyes, jawline) use higher guidance for precision.

    Returns:
        guidance_scale: clamped to [10.0, 14.0] for stability.
    """
    if not features:
        return base_guidance

    total_weight = 0.0
    feature_count = 0

    for key, value in features.items():
        if value and value != "None" and key in FEATURE_GUIDANCE_WEIGHTS:
            total_weight += FEATURE_GUIDANCE_WEIGHTS[key]
            feature_count += 1

    if feature_count == 0:
        return base_guidance

    avg_weight = total_weight / feature_count
    adjusted_guidance = base_guidance * avg_weight

    # Clamp to stable range
    return max(10.0, min(14.0, adjusted_guidance))


def infer_mask_region_from_edit(edit_instruction: str) -> dict:
    """Infer anatomical region from edit instruction text (Improvement 2.3).

    §2.5 fix: Uses list-of-tuples EDIT_REGION_MAPPING (no duplicate-key loss)
    with word-boundary-aware matching via regex. Compound phrases like
    "nose wider" are checked before single words, and \b boundaries prevent
    spurious matches (e.g. "machine" no longer matches "chin").

    Returns:
        dict with keys:
            - 'region': anatomical region name
            - 'dilate_px': suggested dilation in pixels
            - 'buffer': surrounding region to include
            - 'confidence': 0.0-1.0 how confident the inference is
    """
    import re
    instruction_lower = edit_instruction.lower()

    best_match = None
    best_confidence = 0.0

    for keyword, mapping in EDIT_REGION_MAPPING:
        # §2.5 fix: Use word boundaries to avoid substring false positives
        # (e.g. "machine" should NOT match "chin").
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, instruction_lower):
            # Compound phrases (more words / longer strings) get higher confidence
            confidence = min(1.0, 0.5 + (len(keyword) / 15.0))
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = mapping

    if best_match:
        return {
            "region": best_match["region"],
            "dilate_px": best_match["dilate_px"],
            "buffer": best_match["buffer"],
            "confidence": best_confidence,
        }

    # Default to full face if no match
    return {
        "region": "full_face",
        "dilate_px": 0,
        "buffer": "none",
        "confidence": 0.0,
    }


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
    
    # Sentence 2: Structure (with ethnicity-specific anatomical grounding - Improvement 1.5)
    face_shape = features.get("Face shape", "")
    jawline = features.get("Jawline", "")
    struct_parts = []
    if skin and skin != "None": struct_parts.append(f"{skin.lower()} skin")
    if face_shape and face_shape != "None": struct_parts.append(f"an {face_shape.lower()} face shape")
    if jawline and jawline != "None": struct_parts.append(f"a {jawline.lower()} jawline")

    # Add ethnicity-specific anatomical descriptors for coherence
    ethnicity_boost = _get_ethnicity_anatomical_boost(ethnicity)
    if ethnicity_boost:
        struct_parts.append(ethnicity_boost)

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
    """Assemble a forensic-quality SD prompt from individual facial features.

    Incorporates:
    - Ethnicity-specific anatomical grounding (Improvement 1.5)
    - Adaptive guidance scale info in metadata (Improvement 1.3)
    """

    # Generate grammatical narrative
    sentences = _build_narrative(features, is_photo=False)

    # Prefix medium
    prompt = f"A highly detailed front facing {style.lower()} {sentences[0]} {' '.join(sentences[1:])}"

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

# ─── §1.3 fix: Token count guard ────────────────────────────────────────────
# SDXL's CLIP encoders truncate at 77 tokens per encoder. Prompts routinely
# exceed this limit when many features are selected, silently dropping the
# last tokens (often the distinguishing marks and quality boosters appended
# at the end — ironic given their importance). This function estimates the
# token count and trims low-priority suffixes to stay within budget.
_SDXL_TOKEN_LIMIT = 75  # leave 2-token margin for BOS/EOS

def _estimate_token_count(text: str) -> int:
    """Rough CLIP token estimate: ~0.75 tokens per word, commas as separators."""
    words = text.replace(",", " ").split()
    return max(1, int(len(words) * 0.75))

def _trim_prompt_to_budget(prompt: str, budget: int = _SDXL_TOKEN_LIMIT) -> str:
    """Trim prompt from the END (lowest-priority tokens) to fit within budget.

    §1.3 fix: Quality boosters are now prepended before extra_details so they
    are not the first to be truncated. This function trims only if the prompt
    exceeds the budget, removing trailing comma-separated phrases.
    """
    if _estimate_token_count(prompt) <= budget:
        return prompt
    # Split on comma-separated phrases and rebuild until budget is reached
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    trimmed = []
    running_count = 0
    for part in parts:
        part_tokens = _estimate_token_count(part)
        if running_count + part_tokens > budget:
            break
        trimmed.append(part)
        running_count += part_tokens
    return ", ".join(trimmed)


def build_sdxl_forensic_prompt(
    features: dict[str, str],
    style: str = "Pencil sketch",
    extra_details: str = "",
) -> tuple[str, str]:
    """Wrapper that calls build_forensic_prompt and appends SDXL boosters.

    §1.3 fix: Quality boosters are appended BEFORE extra_details so they survive
    token truncation, and the final prompt is trimmed to the CLIP token budget.
    """
    prompt, neg = build_forensic_prompt(features, style, extra_details)
    prompt = prompt + ", best quality, masterpiece, highly detailed face"
    prompt = _trim_prompt_to_budget(prompt)
    return prompt, neg


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
        "hyper-realistic front facing face, highly detailed skin texture, pores, 8k resolution, "
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

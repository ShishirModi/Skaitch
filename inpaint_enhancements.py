# File: inpaint_enhancements.py
# Purpose: Advanced inpainting techniques for improved iterative editing
# Implements improvements 2.1 (Mask Feathering), 2.3 (Prompt-Based Mask Inference),
# and 2.4 (Iterative Refinement with Difference Blending)

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple


# ─── Improvement 2.1: Mask Feathering and Graduated Strength ───────────────────

def feather_mask(mask_pil: Image.Image, sigma: int = 10) -> Image.Image:
    """Apply Gaussian blur feathering to mask edges for seamless blending.

    This creates soft edges on the mask instead of hard boundaries, which eliminates
    visible seams where edited regions meet the surrounding sketch.

    Args:
        mask_pil: Binary mask image (0 or 255, grayscale).
        sigma: Standard deviation for Gaussian blur. Higher = softer edges.

    Returns:
        Feathered mask as PIL Image.
    """
    mask_np = np.array(mask_pil, dtype=np.float32)

    # Apply Gaussian blur to create soft edges
    feathered = cv2.GaussianBlur(mask_np, ksize=(sigma * 2 + 1, sigma * 2 + 1), sigmaX=sigma)

    # Normalize back to 0-255 range
    feathered = (feathered / feathered.max() * 255).astype(np.uint8)

    return Image.fromarray(feathered, mode="L")


def create_graduated_strength_map(
    mask_pil: Image.Image, center_strength: float = 0.85, edge_strength: float = 0.45
) -> np.ndarray:
    """Create a strength map that varies from center (high) to edges (low).

    This allows the masked region's center to undergo significant changes while
    edges blend smoothly into surrounding context.

    Args:
        mask_pil: Binary or feathered mask image (grayscale, 0-255).
        center_strength: Denoise strength at mask center.
        edge_strength: Denoise strength at mask edges.

    Returns:
        Strength map (numpy array, same size as mask, values 0.0-1.0).
    """
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0

    # Normalize mask values to [edge_strength, center_strength]
    strength_map = edge_strength + mask_np * (center_strength - edge_strength)

    return strength_map


def apply_graduated_strength_to_image(
    image_pil: Image.Image,
    original_pil: Image.Image,
    strength_map: np.ndarray,
) -> Image.Image:
    """Blend edited and original image using a graduated strength map.

    Instead of passing strength_map directly to the pipeline (which may not support it),
    we simulate it by blending: result = original * (1 - strength) + edited * strength.

    Args:
        image_pil: Edited image (output from inpainting).
        original_pil: Original sketch image.
        strength_map: Strength map from create_graduated_strength_map().

    Returns:
        Blended image as PIL Image.
    """
    img_np = np.array(image_pil, dtype=np.float32) / 255.0
    orig_np = np.array(original_pil, dtype=np.float32) / 255.0

    # Expand strength_map to 3D if image is RGB
    if len(img_np.shape) == 3:
        strength_expanded = np.stack([strength_map] * 3, axis=-1)
    else:
        strength_expanded = strength_map

    # Blend: higher strength_map = more edited image, lower = more original
    blended = orig_np * (1 - strength_expanded) + img_np * strength_expanded

    # Clip and convert back to uint8
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(blended)


# ─── Improvement 2.3: Prompt-Based Mask Augmentation ──────────────────────────

def suggest_mask_region_from_edit(edit_instruction: str) -> Dict:
    """Suggest mask region based on edit instruction text.

    This is a wrapper around prompt_builder.infer_mask_region_from_edit().
    Returns anatomical suggestions for UI to display to operator.

    Args:
        edit_instruction: Free-text edit instruction (e.g., "make the nose sharper").

    Returns:
        dict with 'region', 'dilate_px', 'buffer', 'confidence' keys.
    """
    from prompt_builder import infer_mask_region_from_edit

    return infer_mask_region_from_edit(edit_instruction)


def auto_dilate_mask(mask_pil: Image.Image, dilate_px: int, kernel_shape: str = "ellipse") -> Image.Image:
    """Dilate mask region by specified pixels.

    Useful for expanding a user-drawn mask to include anatomically-relevant buffer regions.

    Args:
        mask_pil: Binary mask image.
        dilate_px: Number of pixels to dilate.
        kernel_shape: "ellipse", "rect", or "cross".

    Returns:
        Dilated mask as PIL Image.
    """
    if dilate_px <= 0:
        return mask_pil

    mask_np = np.array(mask_pil, dtype=np.uint8)

    # Create morphological kernel
    if kernel_shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    elif kernel_shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    else:  # "cross"
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (dilate_px * 2 + 1, dilate_px * 2 + 1))

    dilated = cv2.dilate(mask_np, kernel, iterations=1)

    return Image.fromarray(dilated, mode="L")


# ─── Improvement 2.4: Iterative Refinement with Difference Blending ──────────

def compute_pixel_difference_magnitude(image1: Image.Image, image2: Image.Image) -> np.ndarray:
    """Compute pixel-wise absolute difference between two images.

    Returns a "change magnitude" map where high values indicate large changes.

    Args:
        image1: Original image.
        image2: Edited image.

    Returns:
        Difference magnitude map (numpy array, 0-255).
    """
    img1_np = np.array(image1, dtype=np.float32)
    img2_np = np.array(image2, dtype=np.float32)

    # Compute absolute difference
    diff = np.abs(img1_np - img2_np)

    # If color image, compute mean across channels
    if len(diff.shape) == 3:
        diff = np.mean(diff, axis=2)

    return diff.astype(np.uint8)


def adaptive_difference_blending(
    original: Image.Image,
    edited: Image.Image,
    max_change_factor: float = 0.85,
) -> Image.Image:
    """Blend original and edited images based on change magnitude.

    §2.3 fix: The original formula computed change_factor = 1.0 - clip(diff, 0, 0.6),
    which meant large deliberate edits (high diff) got a LOW change_factor (≥0.4),
    keeping mostly the original. After 3-4 iterations, cumulative suppression
    compounded and the sketch drifted back to the original despite deliberate changes.

    The new formula uses a soft sigmoid-style curve: moderate changes pass through
    mostly intact, while only extreme outlier changes (>max_change_factor) are
    attenuated. This preserves deliberate structural edits while still preventing
    catastrophic hallucination drift.

    Args:
        original: Original sketch image.
        edited: Edited output from inpainting.
        max_change_factor: Threshold above which changes are attenuated (0.0-1.0).
                          Set to 0.65 in sketch_refiner.py — tight enough to suppress
                          hallucinated deformations while letting deliberate masked edits
                          pass through. The upstream mask compositing step already hard-
                          anchors unmasked regions, so this only acts on the masked zone.

    Returns:
        Blended image as PIL Image.
    """
    orig_np = np.array(original, dtype=np.float32) / 255.0
    edited_np = np.array(edited, dtype=np.float32) / 255.0

    # Compute per-pixel change magnitude
    diff_map = compute_pixel_difference_magnitude(original, edited)
    diff_normalized = diff_map / 255.0

    # §2.3 fix: Soft attenuation curve — changes below threshold pass through
    # at full strength; changes above are smoothly attenuated rather than
    # hard-clipped. This preserves 85%+ of deliberate structural edits.
    # keep_factor: 1.0 where diff < threshold, smoothly decreasing above.
    keep_factor = np.where(
        diff_normalized <= max_change_factor,
        np.ones_like(diff_normalized),
        max_change_factor / (diff_normalized + 1e-6)
    )
    keep_factor = np.clip(keep_factor, 0.4, 1.0)  # floor at 40% to avoid total suppression

    # Expand to 3D for RGB blending
    if len(orig_np.shape) == 3:
        keep_factor = np.stack([keep_factor] * 3, axis=-1)

    # Blend: result = original * (1 - keep_factor) + edited * keep_factor
    blended = orig_np * (1 - keep_factor) + edited_np * keep_factor

    # Clip and convert back to uint8
    blended = (np.clip(blended, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(blended)


def compute_adaptive_inpaint_strength(
    mask_pil: Image.Image,
    base_strength: float = 0.85,
    min_strength: float = 0.70,
    max_strength: float = 0.90,
) -> float:
    """Recommend inpaint strength based on mask size (Improvement 2.2).

    Small masks (< 5%): Can afford high strength for precision.
    Medium masks (5-20%): Balanced strength.
    Large masks (> 20%): Lower strength to preserve surrounding context.

    Args:
        mask_pil: Binary mask image.
        base_strength: Default strength value.
        min_strength: Minimum allowed strength.
        max_strength: Maximum allowed strength.

    Returns:
        Recommended strength value.
    """
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
    mask_area_ratio = np.mean(mask_np)  # 0.0-1.0

    if mask_area_ratio < 0.05:
        # Small mask: high strength for precision
        recommended = max_strength
    elif mask_area_ratio < 0.20:
        # Medium mask: balanced
        recommended = base_strength
    else:
        # Large mask: lower strength
        recommended = min_strength

    return np.clip(recommended, min_strength, max_strength)


# ─── Combined Inpaint Workflow ─────────────────────────────────────────────────

def prepare_enhanced_inpaint_inputs(
    original_sketch: Image.Image,
    user_mask: Image.Image,
    edit_instruction: str,
    apply_feathering: bool = True,
    infer_dilation: bool = True,
) -> Dict:
    """Prepare mask and recommendations for enhanced inpainting workflow.

    Combines improvements 2.1 (feathering), 2.2 (strength auto-tuning),
    and 2.3 (region inference).

    Args:
        original_sketch: Original sketch image.
        user_mask: User-drawn binary mask.
        edit_instruction: Edit instruction text.
        apply_feathering: Whether to feather mask edges.
        infer_dilation: Whether to infer and apply mask dilation.

    Returns:
        dict with processed mask and recommendations:
            - 'mask': feathered/dilated mask for inpainting
            - 'strength': recommended inpaint strength
            - 'strength_map': graduated strength map for blending
            - 'region_suggestion': inferred anatomical region
            - 'region_confidence': confidence in inference (0.0-1.0)
    """
    # Step 1: Feather mask if requested
    if apply_feathering:
        mask = feather_mask(user_mask, sigma=10)
    else:
        mask = user_mask

    # Step 2: Infer region and suggest dilation
    region_info = suggest_mask_region_from_edit(edit_instruction)

    # Step 3: Apply dilation if confidence is high enough
    if infer_dilation and region_info["confidence"] > 0.4:
        mask = auto_dilate_mask(mask, dilate_px=region_info["dilate_px"])

    # Step 4: Create graduated strength map
    strength_map = create_graduated_strength_map(mask)

    # Step 5: Auto-tune inpaint strength based on mask size
    recommended_strength = compute_adaptive_inpaint_strength(mask)

    return {
        "mask": mask,
        "strength": recommended_strength,
        "strength_map": strength_map,
        "region_suggestion": region_info["region"],
        "region_confidence": region_info["confidence"],
    }

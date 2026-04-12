# File: refinement_enhancements.py
# Purpose: Advanced refinement techniques for Phase II photorealistic synthesis
# Implements improvements 3.1 (Adaptive ControlNet Conditioning Scale) and
# 3.5 (Region-Specific Guidance Scale)

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional


# ─── Improvement 3.1: Adaptive ControlNet Conditioning Scale ──────────────────

def compute_edge_contrast(edge_image: np.ndarray) -> float:
    """Compute edge contrast/sharpness from edge detection output.

    §3.2 fix: The original implementation assumed a binary Canny image (0 or 255).
    However, fused_edge_detection() returns a weighted-sum float image with a broad
    range of grey values. The formula now handles both binary and continuous edge maps
    by normalizing the variance computation and using mean edge intensity instead of
    raw variance, which was systematically low on non-binary inputs.

    Args:
        edge_image: Edge detection output (0-255, binary or continuous grayscale).

    Returns:
        contrast_score: 0.0-1.0 where 1.0 = maximum edge clarity.
    """
    total_pixels = edge_image.size
    if total_pixels == 0:
        return 0.5

    # Threshold-independent edge detection: any pixel > 10 is an edge
    edge_mask = edge_image > 10
    edge_pixels = np.sum(edge_mask)
    edge_ratio = edge_pixels / total_pixels

    edges_only = edge_image[edge_mask].astype(np.float32)
    if len(edges_only) > 0:
        # §3.2 fix: Use normalized mean intensity as sharpness proxy.
        # For binary images (Canny), mean ≈ 255 → high score.
        # For continuous images (fused), mean reflects edge strength.
        mean_intensity = np.mean(edges_only) / 255.0
        # Variance normalized to [0, 1] range for continuous images
        variance = np.var(edges_only) / (255.0 * 255.0)
        sharpness = mean_intensity * 0.7 + (1.0 - variance) * 0.3
    else:
        sharpness = 0.0

    # Combine edge coverage and sharpness
    contrast_score = edge_ratio * 0.4 + sharpness * 0.6

    return float(np.clip(contrast_score, 0.0, 1.0))


def compute_adaptive_controlnet_scale(
    canny_image: np.ndarray,
    use_user_preference: bool = False,
    user_preference: str = "balanced",
    base_scale: float = 0.65,
) -> float:
    """Compute adaptive ControlNet conditioning scale based on sketch fidelity.

    If sketch has HIGH-CONTRAST SHARP edges: use scale=0.75 (strict geometry).
    If sketch has SOFT, BLENDED edges: use scale=0.60 (allow hallucination).
    User can override with preference ("strict" vs. "loose").

    Args:
        canny_image: Canny edge detection output.
        use_user_preference: Whether to respect user override.
        user_preference: "strict" (high fidelity) or "loose" (photorealism freedom).
        base_scale: Default scale (typically 0.65).

    Returns:
        controlnet_conditioning_scale: 0.0-1.0 value for ControlNet pipeline.
    """
    # Compute edge contrast
    contrast = compute_edge_contrast(canny_image)

    if use_user_preference:
        if user_preference == "strict":
            return 0.80
        elif user_preference == "loose":
            return 0.55
        # else: "balanced" falls through to adaptive

    # Adaptive scaling based on contrast
    if contrast > 0.7:
        # Sharp, clean edges: strict adherence
        scale = 0.75
    elif contrast > 0.5:
        # Moderate edges: balanced
        scale = 0.65
    else:
        # Soft, blended edges: allow more freedom
        scale = 0.55

    return scale


# ─── Improvement 3.5: Region-Specific Guidance Scale ────────────────────────────

class RegionalGuidanceScaler:
    """Applies region-specific guidance scale adjustments.

    Different facial regions have different importance for perceptual quality:
    - Eyes & mouth (high importance): guidance_scale = 10.0 (strict detail)
    - Skin (medium importance): guidance_scale = 8.5 (smooth)
    - Hair (low importance): guidance_scale = 7.0 (artistic freedom)

    Since diffusers may not support spatial guidance natively, we offer:
    1. Metadata for custom implementations
    2. Post-processing sharpening for critical regions
    """

    # Region definitions (approximate bounding box ratios for 1024x1024 face)
    REGIONS = {
        "eyes": {
            "bbox_ratio": [(0.25, 0.25), (0.75, 0.45)],  # (x_min, y_min), (x_max, y_max)
            "guidance_scale": 10.0,
            "post_process": "sharpen",
        },
        "mouth": {
            "bbox_ratio": [(0.3, 0.60), (0.7, 0.75)],
            "guidance_scale": 10.0,
            "post_process": "sharpen",
        },
        "skin": {
            "bbox_ratio": [(0.1, 0.3), (0.9, 0.85)],
            "guidance_scale": 8.5,
            "post_process": "none",
        },
        "hair": {
            "bbox_ratio": [(0.0, 0.0), (1.0, 0.3)],
            "guidance_scale": 7.0,
            "post_process": "none",
        },
        "neck": {
            "bbox_ratio": [(0.2, 0.8), (0.8, 1.0)],
            "guidance_scale": 7.5,
            "post_process": "none",
        },
    }

    @staticmethod
    def get_region_guidance_map(image_height: int, image_width: int) -> Dict[str, np.ndarray]:
        """Generate guidance scale map for each region.

        Returns a dict of region masks showing where each guidance scale applies.

        Args:
            image_height: Height of image.
            image_width: Width of image.

        Returns:
            dict mapping region_name -> binary mask (numpy array).
        """
        masks = {}

        for region_name, region_cfg in RegionalGuidanceScaler.REGIONS.items():
            mask = np.zeros((image_height, image_width), dtype=np.float32)

            (x_min_ratio, y_min_ratio), (x_max_ratio, y_max_ratio) = region_cfg["bbox_ratio"]
            x_min = int(x_min_ratio * image_width)
            y_min = int(y_min_ratio * image_height)
            x_max = int(x_max_ratio * image_width)
            y_max = int(y_max_ratio * image_height)

            mask[y_min:y_max, x_min:x_max] = 1.0

            masks[region_name] = mask

        return masks

    @staticmethod
    def get_region_guidance_scales() -> Dict[str, float]:
        """Return guidance scales for each region.

        Returns:
            dict mapping region_name -> guidance_scale value.
        """
        return {region: cfg["guidance_scale"] for region, cfg in RegionalGuidanceScaler.REGIONS.items()}

    @staticmethod
    def sharpen_region(image: Image.Image, region_mask: np.ndarray, strength: float = 1.5) -> Image.Image:
        """Apply unsharp masking to a specific region.

        Increases local contrast for eyes/mouth (high perceptual importance).

        Args:
            image: PIL Image to sharpen.
            region_mask: Binary mask indicating region (0.0-1.0).
            strength: Sharpening strength (1.0 = normal, > 1.0 = stronger).

        Returns:
            Sharpened image as PIL Image.
        """
        img_np = np.array(image, dtype=np.float32) / 255.0

        # Create a blurred version
        img_uint8 = (img_np * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, ksize=(5, 5), sigmaX=1.0)
        blurred = blurred.astype(np.float32) / 255.0

        # Unsharp mask: original + (original - blurred) * strength
        sharpened = img_np + (img_np - blurred) * strength

        # Apply only to region (blend with original)
        if len(region_mask.shape) == 2:
            region_mask_3d = np.stack([region_mask] * 3, axis=-1)
        else:
            region_mask_3d = region_mask

        result = img_np * (1 - region_mask_3d) + sharpened * region_mask_3d

        # Clip and convert back to uint8
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        return Image.fromarray(result)

    @staticmethod
    def apply_regional_sharpening(image: Image.Image) -> Image.Image:
        """Apply post-generation sharpening to high-importance regions.

        Enhances eyes and mouth for better perceptual quality.

        Args:
            image: Generated photorealistic image.

        Returns:
            Sharpened image as PIL Image.
        """
        height, width = image.size[1], image.size[0]
        masks = RegionalGuidanceScaler.get_region_guidance_map(height, width)

        result = image

        # Sharpen eyes
        if "eyes" in masks:
            result = RegionalGuidanceScaler.sharpen_region(result, masks["eyes"], strength=1.3)

        # Sharpen mouth
        if "mouth" in masks:
            result = RegionalGuidanceScaler.sharpen_region(result, masks["mouth"], strength=1.2)

        return result


# ─── Integration Helpers ──────────────────────────────────────────────────────

def get_refinement_config(
    canny_image: np.ndarray,
    user_fidelity_preference: str = "balanced",
    apply_post_sharpening: bool = True,
) -> Dict:
    """Generate complete refinement configuration with adaptive parameters.

    Combines improvements 3.1 and 3.5.

    Args:
        canny_image: Canny edge detection output from Phase I sketch.
        user_fidelity_preference: "strict" (geometry lock) or "loose" (photorealism) or "balanced".
        apply_post_sharpening: Whether to apply region-specific sharpening post-generation.

    Returns:
        dict with complete Phase II configuration:
            - 'controlnet_conditioning_scale': adaptive scale based on sketch quality
            - 'guidance_scale': base guidance (fixed at 9.0 in Phase II)
            - 'region_guidance_scales': per-region guidance values (for custom implementations)
            - 'apply_post_sharpening': whether to sharpen eyes/mouth after generation
            - 'num_inference_steps': recommended steps (30)
    """
    use_pref = user_fidelity_preference != "balanced"
    controlnet_scale = compute_adaptive_controlnet_scale(
        canny_image,
        use_user_preference=use_pref,
        user_preference=user_fidelity_preference,
        base_scale=0.65,
    )

    region_guidance = RegionalGuidanceScaler.get_region_guidance_scales()

    return {
        "controlnet_conditioning_scale": controlnet_scale,
        "guidance_scale": 9.0,  # Phase II baseline
        "region_guidance_scales": region_guidance,
        "apply_post_sharpening": apply_post_sharpening,
        "num_inference_steps": 30,
    }


# ─── Advanced Edge Detection (Fix for Edge Detection Limitations) ─────────────

def adaptive_canny_threshold(sketch: np.ndarray) -> Tuple[int, int]:
    """Select appropriate Canny thresholds based on sketch contrast (Edge Detection Fix).

    Rationale: Fresh Phase I sketches have high contrast; edited sketches have lower.
    This function adapts the threshold to maintain edge quality across editing iterations.

    Args:
        sketch: Grayscale sketch image (numpy array, 0-255).

    Returns:
        (low_threshold, high_threshold) tuple for cv2.Canny().
    """
    # Convert to uint8 if necessary
    if sketch.dtype != np.uint8:
        sketch = cv2.convertScaleAbs(sketch)

    contrast = np.std(sketch)

    if contrast > 40:
        # High contrast: clean Phase I generation
        return (100, 200)
    elif contrast > 25:
        # Medium contrast: after 1-2 edits
        return (60, 140)
    elif contrast > 15:
        # Low contrast: after 3+ edits or soft pencil
        return (30, 80)
    else:
        # Very low contrast: heavily edited or charcoal
        return (20, 60)


def fused_edge_detection(sketch_pil: Image.Image, mask_pil: Image.Image = None) -> Image.Image:
    """Multi-modal edge detection fusing Canny, Sobel, and Laplacian (Edge Detection Fix).

    Instead of relying on Canny alone (which can be rigid), fuse multiple edge detectors
    for robustness across different sketch states.

    Args:
        sketch_pil: Sketch image as PIL Image.
        mask_pil: Optional mask indicating editing region.

    Returns:
        Fused edge image as PIL Image (grayscale).
    """
    sketch_np = cv2.cvtColor(np.array(sketch_pil), cv2.COLOR_RGB2GRAY)

    # Determine adaptive Canny thresholds
    low_thresh, high_thresh = adaptive_canny_threshold(sketch_np)

    # 1. Canny edge detection (structural edges)
    canny = cv2.Canny(sketch_np, low_thresh, high_thresh)

    # 2. Sobel edge detection (gradient-based, smoother)
    sobelx = cv2.Sobel(sketch_np, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(sketch_np, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold Sobel to get binary edges
    sobel_binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)[1]

    # 3. Laplacian edge detection (fine detail preservation)
    laplacian = cv2.Laplacian(sketch_np, cv2.CV_64F)
    laplacian_abs = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    laplacian_binary = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)[1]

    # Weighted fusion: Canny (0.5) + Sobel (0.3) + Laplacian (0.2)
    # Higher weight on Canny for structural accuracy
    fused = (
        canny.astype(np.float32) * 0.5
        + sobel_binary.astype(np.float32) * 0.3
        + laplacian_binary.astype(np.float32) * 0.2
    ).astype(np.uint8)

    # If mask provided: emphasize mask boundaries for better transitions
    if mask_pil is not None:
        mask_np = np.array(mask_pil, dtype=np.uint8)
        # Extract mask boundary edges
        mask_edges = cv2.Canny(mask_np, 30, 100)
        # Combine with sketch edges: take maximum (union of edges)
        fused = cv2.max(fused, mask_edges)

    return Image.fromarray(fused, mode="L")


def enhance_sketch_for_edge_detection(sketch_pil: Image.Image) -> Image.Image:
    """Enhance sketch contrast before edge detection (Edge Detection Fix).

    Use CLAHE (Contrast Limited Adaptive Histogram Equalization) and unsharp masking
    to improve sketch clarity for better edge detection.

    Args:
        sketch_pil: Sketch image as PIL Image.

    Returns:
        Enhanced sketch as PIL Image.
    """
    sketch_np = cv2.cvtColor(np.array(sketch_pil), cv2.COLOR_RGB2GRAY)

    # Step 1: Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sketch_np)

    # Step 2: Unsharp masking for edge sharpening
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Convert back to RGB for compatibility
    enhanced_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(enhanced_rgb)


def apply_mask_context_awareness(
    edges_pil: Image.Image, mask_pil: Image.Image, dilate_px: int = 15
) -> Image.Image:
    """Emphasize edges around mask boundary for better transition (Edge Detection Fix).

    Enhances the transition zone between masked and unmasked regions to prevent
    visible seams during ControlNet synthesis.

    Args:
        edges_pil: Edge detection output as PIL Image.
        mask_pil: Mask image as PIL Image.
        dilate_px: Dilation radius for boundary extraction.

    Returns:
        Context-aware edge image as PIL Image.
    """
    edges_np = np.array(edges_pil, dtype=np.uint8)
    mask_np = np.array(mask_pil, dtype=np.uint8)

    # Dilate mask to find boundary region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    dilated_mask = cv2.dilate(mask_np, kernel)
    boundary = dilated_mask - mask_np

    # Enhance edges in boundary region (1.3x boost)
    edges_np[boundary > 0] = np.minimum(edges_np[boundary > 0] * 1.3, 255)

    return Image.fromarray(edges_np, mode="L")


# ─── Diagnostics ──────────────────────────────────────────────────────────────

def analyze_canny_quality(canny_image: np.ndarray) -> Dict:
    """Analyze Canny edge output for diagnostic purposes.

    Returns metrics about edge quality, noise, and suggested adjustments.

    Args:
        canny_image: Output from cv2.Canny().

    Returns:
        dict with analysis:
            - 'contrast': edge contrast score (0.0-1.0)
            - 'edge_coverage': % of image that contains edges
            - 'recommended_scale': suggested ControlNet conditioning scale
            - 'quality_rating': "excellent", "good", "moderate", "poor"
    """
    contrast = compute_edge_contrast(canny_image)
    edge_coverage = np.sum(canny_image > 0) / canny_image.size

    recommended_scale = compute_adaptive_controlnet_scale(canny_image)

    if contrast > 0.75:
        quality = "excellent"
    elif contrast > 0.60:
        quality = "good"
    elif contrast > 0.40:
        quality = "moderate"
    else:
        quality = "poor"

    return {
        "contrast": contrast,
        "edge_coverage": edge_coverage,
        "recommended_scale": recommended_scale,
        "quality_rating": quality,
    }

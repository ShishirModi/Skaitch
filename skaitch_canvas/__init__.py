import os
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
import base64
import io

class CanvasResult:
    def __init__(self, image_data):
        self.image_data = image_data

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "skaitch_canvas",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend")
    _component_func = components.declare_component("skaitch_canvas", path=build_dir)

def st_skaitch_canvas(fill_color=None, stroke_width=20, stroke_color="#F59E0B", background_image=None, update_streamlit=True, height=600, width=800, drawing_mode="freedraw", key=None):
    """
    Render the custom Skaitch Canvas natively.
    Accepts a PIL Image for the background, bypassing Streamlit's broken 'image_to_url' cache.
    Returns an object with .image_data containing the RGBA numpy array of drawn strokes.
    """
    bg_image_b64 = None
    if background_image:
        img_buffer = io.BytesIO()
        if background_image.mode != "RGB":
            background_image = background_image.convert("RGB")
        background_image.save(img_buffer, format="JPEG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        bg_image_b64 = f"data:image/jpeg;base64,{img_str}"

    component_value = _component_func(
        bg_image=bg_image_b64,
        brush_size=stroke_width,
        stroke_color=stroke_color,
        width=width,
        height=height,
        key=key,
        default=None
    )
    if component_value:
        try:
            base64_data = component_value.split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(base64_data)))
            # Ensure the image is RGBA to match streamlit-drawable-canvas API downstream
            img = img.convert("RGBA")
            return CanvasResult(np.array(img))
        except Exception as e:
            return CanvasResult(None)
    
    return CanvasResult(None)

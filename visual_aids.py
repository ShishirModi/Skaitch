# File: visual_aids.py
# Purpose: Basic SVG representations of forensic facial features

import base64

def get_svg_html(svg_string: str) -> str:
    """Wraps an SVG string in an HTML div for Streamlit."""
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; 
                background: rgba(255,255,255,0.05); border-radius: 8px; 
                padding: 10px; margin-top: 5px; margin-bottom: 15px; 
                height: 80px; width: 100%;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="60" height="60">
            {svg_string}
        </svg>
    </div>
    """

# Default styles for paths
S_FACE = 'fill="none" stroke="#a78bfa" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"'
S_EYE = 'fill="none" stroke="#a78bfa" stroke-width="2" stroke-linecap="round"'
S_NOSE = 'fill="none" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"'
S_JAW = 'fill="none" stroke="#a78bfa" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"'

VISUAL_AIDS = {
    "Face shape": {
        "Oval": f'<ellipse cx="50" cy="50" rx="35" ry="45" {S_FACE} />',
        "Round": f'<circle cx="50" cy="50" r="40" {S_FACE} />',
        "Square": f'<rect x="15" y="15" width="70" height="70" rx="10" {S_FACE} />',
        "Heart": f'<path d="M 50 85 C 50 85, 10 50, 15 25 C 20 10, 40 10, 50 25 C 60 10, 80 10, 85 25 C 90 50, 50 85, 50 85 Z" {S_FACE} />',
        "Oblong": f'<rect x="20" y="5" width="60" height="90" rx="30" {S_FACE} />',
        "Diamond": f'<path d="M 50 5 L 85 50 L 50 95 L 15 50 Z" {S_FACE} />',
        "Triangle": f'<path d="M 50 15 L 85 85 L 15 85 Z" {S_FACE} />',
        "Pear": f'<path d="M 50 10 C 30 10, 20 40, 15 70 C 10 90, 90 90, 85 70 C 80 40, 70 10, 50 10 Z" {S_FACE} />',
    },
    "Eyes": {
        "Almond": f'<path d="M 20 50 Q 50 20 80 50 Q 50 80 20 50 Z" {S_EYE} /><circle cx="50" cy="50" r="10" {S_EYE} />',
        "Round": f'<path d="M 25 50 A 25 25 0 0 1 75 50 A 25 25 0 0 1 25 50" {S_EYE} /><circle cx="50" cy="50" r="12" {S_EYE} />',
        "Hooded": f'<path d="M 20 55 Q 50 35 80 55 Q 50 80 20 55 Z" {S_EYE} /><path d="M 15 45 Q 50 15 85 45" {S_EYE} /><circle cx="50" cy="55" r="9" {S_EYE} />',
        "Deep-set": f'<path d="M 25 55 Q 50 25 75 55 Q 50 85 25 55 Z" {S_EYE} /><path d="M 20 35 Q 50 15 80 35" stroke-width="4" stroke="#a78bfa" fill="none" /><circle cx="50" cy="55" r="9" {S_EYE} />',
        "Monolid": f'<path d="M 20 50 Q 50 35 80 50 Q 50 65 20 50 Z" {S_EYE} /><circle cx="50" cy="50" r="8" {S_EYE} />',
        "Wide-set": f'<path d="M 10 50 Q 25 35 40 50 Q 25 65 10 50 Z" {S_EYE} /><circle cx="25" cy="50" r="6" {S_EYE} /><path d="M 60 50 Q 75 35 90 50 Q 75 65 60 50 Z" {S_EYE} /><circle cx="75" cy="50" r="6" {S_EYE} />',
        "Close-set": f'<path d="M 25 50 Q 40 30 45 50 Q 40 70 25 50 Z" {S_EYE} /><circle cx="35" cy="50" r="6" {S_EYE} /><path d="M 55 50 Q 60 30 75 50 Q 60 70 55 50 Z" {S_EYE} /><circle cx="65" cy="50" r="6" {S_EYE} />',
        "Upturned": f'<path d="M 20 60 Q 50 30 85 40 Q 50 80 20 60 Z" {S_EYE} /><circle cx="52" cy="52" r="9" {S_EYE} />',
        "Downturned": f'<path d="M 15 40 Q 50 30 80 60 Q 50 80 15 40 Z" {S_EYE} /><circle cx="48" cy="52" r="9" {S_EYE} />',
    },
    "Nose": {
        "Straight": f'<path d="M 50 20 L 50 70 M 35 70 Q 50 85 65 70" {S_NOSE} />',
        "Broad": f'<path d="M 50 20 L 50 60 M 25 70 Q 50 90 75 70" {S_NOSE} />',
        "Narrow": f'<path d="M 50 20 L 50 75 M 42 75 Q 50 85 58 75" {S_NOSE} />',
        "Aquiline": f'<path d="M 50 20 Q 70 45 50 70 M 35 70 Q 50 85 65 70" {S_NOSE} />',
        "Button": f'<path d="M 50 30 L 50 60 M 35 65 Q 50 80 65 65 M 50 60 A 5 5 0 1 1 50 70 A 5 5 0 1 1 50 60" {S_NOSE} />',
        "Wide bridge": f'<path d="M 40 20 L 40 60 M 60 20 L 60 60 M 30 70 Q 50 85 70 70" {S_NOSE} />',
        "Snub": f'<path d="M 50 30 Q 35 50 50 65 M 35 70 Q 50 75 65 70 M 50 70 L 50 65" {S_NOSE} />',
        "Roman": f'<path d="M 50 15 Q 65 30 50 50 L 50 75 M 35 75 Q 50 90 65 75" {S_NOSE} />',
        "Bulbous": f'<path d="M 50 20 L 50 50 M 30 70 Q 50 100 70 70 M 50 50 A 15 15 0 1 1 50 80 A 15 15 0 1 1 50 50" {S_NOSE} />',
        "Hawk": f'<path d="M 50 20 Q 80 40 50 80 M 35 75 Q 50 85 65 75" {S_NOSE} />',
    },
    "Jawline": {
        "Strong": f'<path d="M 20 20 L 20 60 L 40 85 L 60 85 L 80 60 L 80 20" {S_JAW} />',
        "Soft": f'<path d="M 20 20 Q 20 90 50 90 Q 80 90 80 20" {S_JAW} />',
        "Pointed": f'<path d="M 20 20 L 30 60 L 50 95 L 70 60 L 80 20" {S_JAW} />',
        "Wide": f'<path d="M 10 20 L 10 70 L 40 90 L 60 90 L 90 70 L 90 20" {S_JAW} />',
        "Receding": f'<path d="M 30 20 L 30 50 Q 50 70 70 50 L 70 20" {S_JAW} />',
        "V-shaped": f'<path d="M 20 20 L 35 60 L 50 95 L 65 60 L 80 20" {S_JAW} />',
    }
}

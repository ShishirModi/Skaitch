# File: visual_aids.py
# Purpose: Basic SVG representations of forensic facial features

import base64

def get_svg_html(svg_string: str) -> str:
    """Wraps an SVG string in an HTML div for Streamlit."""
    return f"""
    <div style="display: flex; justify-content: center; align-items: center; 
                background: #FFFFFF; border: 1px solid rgba(0,0,0,0.05); 
                border-radius: 6px; padding: 10px; margin-top: 4px; margin-bottom: 12px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                height: auto; min-height: 70px; max-height: 90px; width: 100%;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%" style="max-width: 65px; max-height: 65px;">
            {svg_string}
        </svg>
    </div>
    """

# Default styles for two-tone paths (Stripe SaaS Light Theme)
S_BASE = 'fill="none" stroke="rgba(28,28,30,0.95)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"'
S_HL = 'fill="none" stroke="rgba(245,158,11,1.0)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"'
F_EYE = 'fill="rgba(28,28,30,0.15)"' # Increased opacity for visibility

VISUAL_AIDS = {
    "Face shape": {
        "Oval": f'<ellipse cx="50" cy="50" rx="35" ry="45" {S_BASE} /><path d="M 17 65 Q 50 100 83 65" {S_HL} />',
        "Round": f'<circle cx="50" cy="50" r="40" {S_BASE} /><path d="M 15 70 Q 50 100 85 70" {S_HL} />',
        "Square": f'<rect x="20" y="20" width="60" height="60" rx="15" {S_BASE} /><path d="M 20 55 L 20 65 Q 20 80 35 80 L 65 80 Q 80 80 80 65 L 80 55" {S_HL} />',
        "Heart": f'<path d="M 50 85 C 50 85, 10 50, 15 25 C 20 10, 40 10, 50 25 C 60 10, 80 10, 85 25 C 90 50, 50 85, 50 85 Z" {S_BASE} /><path d="M 18 55 Q 50 90 82 55" {S_HL} />',
        "Oblong": f'<rect x="25" y="10" width="50" height="80" rx="25" {S_BASE} /><path d="M 25 65 Q 50 100 75 65" {S_HL} />',
        "Diamond": f'<path d="M 50 10 L 85 50 L 50 90 L 15 50 Z" {S_BASE} /><path d="M 25 60 L 50 90 L 75 60" {S_HL} />',
        "Triangle": f'<path d="M 50 15 L 85 85 L 15 85 Z" {S_BASE} /><path d="M 25 70 L 50 25 L 75 70" {S_HL} />',
        "Pear": f'<path d="M 50 10 C 35 10, 25 40, 20 65 C 15 90, 85 90, 80 65 C 75 40, 65 10, 50 10 Z" {S_BASE} /><path d="M 22 55 C 20 95, 80 95, 78 55" {S_HL} />',
    },
    "Eyes": {
        "Almond": f'<path d="M 20 50 Q 50 25 80 50 Q 50 75 20 50 Z" {S_BASE} /><path d="M 20 50 Q 50 25 80 50" {S_HL} /><circle cx="50" cy="50" r="10" {S_BASE} {F_EYE} /><circle cx="50" cy="50" r="3" fill="rgba(245,158,11,0.85)" />',
        "Round": f'<path d="M 25 50 A 25 25 0 0 1 75 50 A 25 25 0 0 1 25 50" {S_BASE} /><path d="M 25 50 A 25 25 0 0 1 75 50" {S_HL} /><circle cx="50" cy="50" r="12" {S_BASE} {F_EYE} /><circle cx="50" cy="50" r="3" fill="rgba(245,158,11,0.85)" />',
        "Hooded": f'<path d="M 20 55 Q 50 40 80 55 Q 50 75 20 55 Z" {S_BASE} /><path d="M 15 45 Q 50 20 85 45" {S_HL} /><circle cx="50" cy="55" r="9" {S_BASE} {F_EYE} /><circle cx="50" cy="55" r="3" fill="rgba(245,158,11,0.85)" />',
        "Deep-set": f'<path d="M 25 55 Q 50 30 75 55 Q 50 80 25 55 Z" {S_BASE} /><path d="M 20 35 Q 50 15 80 35" {S_HL} /><circle cx="50" cy="55" r="9" {S_BASE} {F_EYE} /><circle cx="50" cy="55" r="3" fill="rgba(245,158,11,0.85)" />',
        "Monolid": f'<path d="M 20 50 Q 50 40 80 50 Q 50 60 20 50 Z" {S_BASE} /><path d="M 20 50 Q 50 40 80 50" {S_HL} /><circle cx="50" cy="50" r="8" {S_BASE} {F_EYE} /><circle cx="50" cy="50" r="2.5" fill="rgba(245,158,11,0.85)" />',
        "Wide-set": f'<path d="M 10 50 Q 25 35 40 50 Q 25 65 10 50 Z" {S_BASE} /><path d="M 10 50 Q 25 35 40 50" {S_HL} /><circle cx="25" cy="50" r="6" {S_BASE} {F_EYE} /><circle cx="25" cy="50" r="2.5" fill="rgba(245,158,11,0.85)" /><path d="M 60 50 Q 75 35 90 50 Q 75 65 60 50 Z" {S_BASE} /><path d="M 60 50 Q 75 35 90 50" {S_HL} /><circle cx="75" cy="50" r="6" {S_BASE} {F_EYE} /><circle cx="75" cy="50" r="2.5" fill="rgba(245,158,11,0.85)" />',
        "Close-set": f'<path d="M 25 50 Q 37 35 50 50 Q 37 65 25 50 Z" {S_BASE} /><path d="M 25 50 Q 37 35 50 50" {S_HL} /><circle cx="37" cy="50" r="7" {S_BASE} {F_EYE} /><circle cx="37" cy="50" r="2.5" fill="rgba(245,158,11,0.85)" /><path d="M 50 50 Q 63 35 75 50 Q 63 65 50 50 Z" {S_BASE} /><path d="M 50 50 Q 63 35 75 50" {S_HL} /><circle cx="63" cy="50" r="7" {S_BASE} {F_EYE} /><circle cx="63" cy="50" r="2.5" fill="rgba(245,158,11,0.85)" />',
        "Upturned": f'<path d="M 20 60 Q 50 35 85 45 Q 50 80 20 60 Z" {S_BASE} /><path d="M 20 60 Q 50 35 85 45" {S_HL} /><circle cx="52" cy="54" r="9" {S_BASE} {F_EYE} /><circle cx="52" cy="54" r="3" fill="rgba(245,158,11,0.85)" />',
        "Downturned": f'<path d="M 15 45 Q 50 35 80 60 Q 50 80 15 45 Z" {S_BASE} /><path d="M 15 45 Q 50 35 80 60" {S_HL} /><circle cx="48" cy="54" r="9" {S_BASE} {F_EYE} /><circle cx="48" cy="54" r="3" fill="rgba(245,158,11,0.85)" />',
    },
    "Nose": {
        "Straight": f'<path d="M 45 20 L 45 65 M 55 20 L 55 65" {S_BASE} /><path d="M 35 70 Q 50 80 65 70" {S_HL} />',
        "Broad": f'<path d="M 50 20 L 50 63" {S_BASE} /><path d="M 25 70 Q 50 85 75 70" {S_HL} />',
        "Narrow": f'<path d="M 50 20 L 50 75" {S_BASE} /><path d="M 42 75 Q 50 82 58 75" {S_HL} />',
        "Aquiline": f'<path d="M 50 20 Q 65 45 50 70" {S_BASE} /><path d="M 35 70 Q 50 80 65 70" {S_HL} />',
        "Button": f'<path d="M 50 30 L 50 60 M 50 60 A 5 5 0 1 1 50 70 A 5 5 0 1 1 50 60" {S_BASE} /><path d="M 35 65 Q 50 75 65 65" {S_HL} />',
        "Wide bridge": f'<path d="M 40 20 L 40 60 M 60 20 L 60 60" {S_BASE} /><path d="M 30 70 Q 50 80 70 70" {S_HL} />',
        "Snub": f'<path d="M 50 30 Q 38 50 50 65 M 50 70 L 50 65" {S_BASE} /><path d="M 35 70 Q 50 75 65 70" {S_HL} />',
        "Roman": f'<path d="M 50 15 Q 65 30 50 50 L 50 75" {S_BASE} /><path d="M 35 75 Q 50 85 65 75" {S_HL} />',
        "Bulbous": f'<path d="M 50 20 L 50 50 M 50 50 A 15 15 0 1 1 50 80 A 15 15 0 1 1 50 50" {S_BASE} /><path d="M 30 70 Q 50 95 70 70" {S_HL} />',
        "Hawk": f'<path d="M 50 20 Q 75 40 50 80" {S_BASE} /><path d="M 35 75 Q 50 82 65 75" {S_HL} />',
    },
    "Jawline": {
        "Strong": f'<path d="M 20 20 L 20 60 L 40 85 L 60 85 L 80 60 L 80 20" {S_BASE} /><path d="M 20 60 L 40 85 L 60 85 L 80 60" {S_HL} />',
        "Soft": f'<path d="M 20 20 Q 20 90 50 90 Q 80 90 80 20" {S_BASE} /><path d="M 25 70 Q 50 90 75 70" {S_HL} />',
        "Pointed": f'<path d="M 20 20 L 30 60 L 50 95 L 70 60 L 80 20" {S_BASE} /><path d="M 30 60 L 50 95 L 70 60" {S_HL} />',
        "Wide": f'<path d="M 10 20 L 10 70 L 35 90 L 65 90 L 90 70 L 90 20" {S_BASE} /><path d="M 10 70 L 35 90 L 65 90 L 90 70" {S_HL} />',
        "Receding": f'<path d="M 30 20 L 30 50 Q 50 70 70 50 L 70 20" {S_BASE} /><path d="M 30 50 Q 50 70 70 50" {S_HL} />',
        "V-shaped": f'<path d="M 20 20 L 35 55 L 50 90 L 65 55 L 80 20" {S_BASE} /><path d="M 35 55 L 50 90 L 65 55" {S_HL} />',
    }
}

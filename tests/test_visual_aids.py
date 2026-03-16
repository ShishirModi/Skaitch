import unittest
import sys
import os

# Add parent directory to path so we can import from visual_aids and prompt_builder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from prompt_builder import FACIAL_FEATURES
from visual_aids import VISUAL_AIDS

class TestVisualAids(unittest.TestCase):
    def test_svg_keys_match_features(self):
        # The categories we are currently displaying SVGs for in app.py
        svg_categories = ["Face shape", "Jawline", "Eyes", "Nose"]
        
        for category in svg_categories:
            # Check that the category exists in VISUAL_AIDS
            self.assertIn(category, VISUAL_AIDS)
            
            # Check that every option in FACIAL_FEATURES has an SVG in VISUAL_AIDS
            for option in FACIAL_FEATURES[category]:
                self.assertIn(option, VISUAL_AIDS[category], f"Missing SVG for '{category}': '{option}'")

if __name__ == '__main__':
    unittest.main()

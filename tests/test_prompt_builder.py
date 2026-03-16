import unittest
from prompt_builder import build_forensic_prompt, FORENSIC_NEGATIVE

class TestPromptBuilder(unittest.TestCase):
    def test_default_prompt(self):
        prompt, negative = build_forensic_prompt({})
        self.assertIn("Pencil sketch of a male face", prompt)
        self.assertIn("age 26–35", prompt)
        self.assertEqual(negative, FORENSIC_NEGATIVE)
        
    def test_full_features(self):
        features = {
            "Gender": "Female",
            "Age range": "18–25",
            "Face shape": "Oval",
            "Jawline": "V-shaped",
            "Eyes": "Almond",
            "Eyebrows": "High arch",
            "Nose": "Button",
            "Mouth / Lips": "Full",
            "Skin tone": "Medium",
            "Hair style": "Curly",
            "Hair color": "Black",
            "Facial hair": "None",
            "Distinguishing marks": "Freckles",
        }
        prompt, _ = build_forensic_prompt(features, style="Charcoal sketch", extra_details="wearing glasses")
        self.assertIn("Charcoal sketch of a female face", prompt)
        self.assertIn("age 18–25", prompt)
        self.assertIn("oval face shape", prompt)
        self.assertIn("v-shaped jawline", prompt)
        self.assertIn("almond eyes", prompt)
        self.assertIn("high arch eyebrows", prompt)
        self.assertIn("button nose", prompt)
        self.assertIn("full lips", prompt)
        self.assertIn("medium skin tone", prompt)
        self.assertIn("black curly hair", prompt)
        self.assertIn("freckles", prompt)
        self.assertIn("wearing glasses", prompt)
        self.assertNotIn("none", prompt.lower())

    def test_bald_hair(self):
        features = {
            "Hair style": "Bald",
            "Hair color": "Blonde" # Hair color should be ignored if bald
        }
        prompt, _ = build_forensic_prompt(features)
        self.assertIn("bald head", prompt)
        self.assertNotIn("blonde", prompt.lower())

if __name__ == '__main__':
    unittest.main()

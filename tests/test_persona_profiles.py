import json
from pathlib import Path
import tempfile
import unittest

from persona_profiles import compose_prompt, get_persona, load_personas


class PersonaProfileTests(unittest.TestCase):
    def make_profile(self, root, identifier="sample"):
        directory = Path(root) / identifier
        directory.mkdir()
        (directory / "identity.png").write_bytes(b"png")
        (directory / "canon.md").write_text("Keep the silver bob haircut.")
        (directory / "persona.json").write_text(json.dumps({
            "schema_version": 1,
            "name": "Sample Character",
            "description": "Synthetic test preset",
            "prompt_file": "canon.md",
            "references": [{"label": "Identity", "path": "identity.png"}],
        }))
        return directory

    def test_loads_private_profile_and_composes_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_profile(root)
            profiles = load_personas(root)
            self.assertEqual(len(profiles), 1)
            profile = profiles[0]
            self.assertEqual(profile.identifier, "sample")
            self.assertEqual(profile.reference_labels, ("Identity",))
            self.assertEqual(
                compose_prompt(profile, "Standing beside a window."),
                "Keep the silver bob haircut.\n\nStanding beside a window.",
            )

    def test_invalid_profile_does_not_hide_valid_profiles(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_profile(root, "valid")
            broken = Path(root) / "broken"
            broken.mkdir()
            (broken / "persona.json").write_text("{not json")
            self.assertEqual([profile.identifier for profile in load_personas(root)], ["valid"])

    def test_rejects_reference_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            outside = Path(root).parent / "outside.png"
            outside.write_bytes(b"outside")
            self.addCleanup(outside.unlink, missing_ok=True)
            directory = Path(root) / "unsafe"
            directory.mkdir()
            (directory / "persona.json").write_text(json.dumps({
                "schema_version": 1,
                "name": "Unsafe",
                "references": [{"path": "../outside.png"}],
            }))
            self.assertEqual(load_personas(root), [])

    def test_get_persona_rejects_unknown_identifier(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                get_persona("../outside", root)


if __name__ == "__main__":
    unittest.main()

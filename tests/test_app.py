"""Real Gradio callback integration, with only expensive inference substituted."""

import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) for name in ("gradio", "PIL"))


@unittest.skipUnless(RUNTIME_AVAILABLE, "Install requirements-ci.txt for Gradio integration tests")
class AppTests(unittest.TestCase):
    def setUp(self):
        import lab
        from gradio.state_holder import SessionState

        self.lab = lab
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output_patch = patch.object(lab, "OUTPUTS", self.root)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.enhancer_patch = patch.object(lab, "PROMPT_ENHANCER_ENABLED", False)
        self.enhancer_patch.start()
        self.addCleanup(self.enhancer_patch.stop)
        self.app = lab.build_app()
        self.state = SessionState(self.app)
        self.functions = {fn.name: (index, fn.fn) for index, fn in self.app.fns.items()}

    def fake_generate(self, prompt, refs, width, height, steps, seed, context=None,
                      user_prompt=None, progress_callback=None, preview_callback=None,
                      enhancer=None):
        from PIL import Image
        from history import save_generation

        metadata = dict(prompt=prompt, width=width, height=height, steps=steps,
                        seed=seed, elapsed_seconds=1.25, peak_allocated_gib=1.0)
        if user_prompt:
            metadata["user_prompt"] = user_prompt
        if context:
            metadata["context"] = context
        if enhancer:
            metadata["prompt_enhancer"] = enhancer
        return save_generation(self.root, Image.new("RGBA", (width, height), "red"), metadata, refs or [])

    def test_generation_updates_table_gallery_and_selection_snapshot(self):
        with patch.object(self.lab, "generate", self.fake_generate):
            result = asyncio.run(self.app.process_api(
                self.functions["run"][0],
                ["", "Test city", None, "カスタム", 512, 512, 4, 99, False],
                state=self.state,
            ))
        data = result["data"]
        self.assertIn("Test city", str(data[3]))
        self.assertEqual(len(data[4]), 1)
        entries = self.lab.load_history(self.root)
        restored = self.functions["restore"][1](0, [entries[0]["id"]])
        self.assertEqual(restored[:9], ("", "Test city", [], "カスタム", 512, 512, 4, 99, False))
        self.assertTrue(Path(restored[9]).is_file())
        self.assertEqual(len(self.functions["refresh"][1]()[1]), 1)

    def test_empty_history_and_stale_selection(self):
        import gradio as gr

        _, gallery, ids = self.functions["refresh"][1]()
        self.assertEqual((gallery, ids), ([], []))
        with self.assertRaises(gr.Error):
            self.functions["restore"][1](0, [])

    def test_failed_generation_does_not_add_history(self):
        import gradio as gr

        with patch.object(self.lab, "generate", side_effect=ValueError("Enter a prompt.")):
            with self.assertRaises(gr.Error):
                self.functions["run"][1]("", "", None, "カスタム", 512, 512, 4, 0, False)
        self.assertEqual(self.lab.load_history(self.root), [])

    def test_reference_edit_is_retained_and_restored(self):
        from PIL import Image

        reference = self.root / "uploaded-reference.png"
        Image.new("RGB", (64, 64), "blue").save(reference)
        with patch.object(self.lab, "generate", self.fake_generate):
            response = self.functions["run"][1](
                "", "An edited city", [str(reference)], "カスタム", 512, 512, 4, 100, False
            )
        reference.unlink()
        restored = self.functions["restore"][1](0, response[5])
        self.assertEqual(restored[1], "An edited city")
        self.assertEqual(len(restored[2]), 1)
        self.assertTrue(Path(restored[2][0]).is_file())
        self.assertEqual(len(response[3].samples), 1)

    def test_manual_reference_rejects_server_path_outside_allowed_roots(self):
        import gradio as gr

        with tempfile.TemporaryDirectory() as external:
            reference = Path(external) / "server-local.png"
            reference.write_bytes(b"not an upload")
            with self.assertRaises(gr.Error):
                self.functions["run"][1](
                    "", "Unsafe reference", [str(reference)],
                    "カスタム", 512, 512, 4, 100, False,
                )

    def test_manual_reference_accepts_gradio_upload_root(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as uploads:
            reference = Path(uploads) / "browser-upload.png"
            Image.new("RGB", (64, 64), "blue").save(reference)
            with patch.dict(os.environ, {"GRADIO_TEMP_DIR": uploads}):
                self.assertEqual(
                    self.lab.validate_manual_references([str(reference)]),
                    [str(reference.resolve())],
                )

    def test_persona_preset_adds_private_prompt_and_reference(self):
        from PIL import Image
        from persona_profiles import PersonaProfile

        reference = self.root / "persona-identity.png"
        Image.new("RGB", (64, 64), "green").save(reference)
        profile = PersonaProfile(
            identifier="sample",
            name="Sample Character",
            description="Synthetic preset",
            prompt_prefix="Keep the short silver hair.",
            references=(reference,),
            reference_labels=("Identity",),
        )
        with (patch.object(self.lab, "get_persona", return_value=profile),
              patch.object(self.lab, "generate", self.fake_generate)):
            response = self.functions["run"][1](
                "sample", "Standing in a studio.", None,
                "カスタム", 512, 512, 4, 100, False,
            )

        entry = self.lab.load_history(self.root)[0]
        self.assertIn("Keep the short silver hair.", entry["prompt"])
        self.assertIn("Do not render prompt text", entry["prompt"])
        self.assertTrue(entry["prompt"].startswith("Standing in a studio."))
        self.assertEqual(entry["user_prompt"], "Standing in a studio.")
        self.assertEqual(entry["context"]["persona_name"], "Sample Character")
        self.assertEqual(entry["context"]["persona_reference_count"], 1)
        self.assertEqual(len(entry["references"]), 1)
        self.assertEqual(len(response[3].samples), 1)
        restored = self.functions["restore"][1](0, response[5])
        self.assertEqual(restored[1], "Standing in a studio.")

    def test_delete_selected_requires_selection_and_refreshes_history(self):
        with patch.object(self.lab, "generate", self.fake_generate):
            response = self.functions["run"][1](
                "", "Disposable city", None, "カスタム", 512, 512, 4, 101, False
            )
        identifier = response[6]
        confirmation = self.functions["request_delete"][1](identifier)
        self.assertTrue(confirmation.visible)
        deleted = self.functions["delete_selected"][1](identifier)

        self.assertEqual(deleted[:4], (None, None, "", None))
        self.assertEqual(deleted[-2:], ([], []))
        self.assertEqual(self.lab.load_history(self.root), [])

    def test_batch_selection_and_delete_multiple_generations(self):
        with patch.object(self.lab, "generate", self.fake_generate):
            self.functions["run"][1](
                "", "First disposable image", None, "カスタム", 512, 512, 4, 201, False
            )
            self.functions["run"][1](
                "", "Second disposable image", None, "カスタム", 512, 512, 4, 202, False
            )
        identifiers = [entry["id"] for entry in self.lab.load_history(self.root)]
        first = self.functions["select_image"][1](
            identifiers, False, True, [], SimpleNamespace(index=0)
        )
        selected = first[13]
        self.assertEqual(selected, [identifiers[0]])
        second = self.functions["select_image"][1](
            identifiers, False, True, selected, SimpleNamespace(index=1)
        )
        selected = second[13]
        self.assertEqual(set(selected), set(identifiers))

        deleted = self.functions["delete_batch"][1](selected)
        self.assertEqual(deleted[4], [])
        self.assertEqual(self.lab.load_history(self.root), [])

    def test_gallery_selection_is_ignored_while_generation_is_active(self):
        response = self.functions["select_image"][1](
            ["missing-but-must-not-be-restored"], True, False, [], SimpleNamespace(index=0)
        )
        self.assertEqual(response[13], [])

    def test_delete_is_private_and_generate_api_contract_is_unchanged(self):
        endpoints = self.app.get_api_info()["named_endpoints"]
        self.assertEqual(set(endpoints), {"/generate", "/use_reference"})
        self.assertEqual(len(endpoints["/generate"]["returns"]), 5)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from manage import (LEGACY_MODEL_MARKER, MODEL_MARKER, download,
                    download_enhancers, installed_model_revision,
                    migrate_legacy_model_marker, missing_model_files,
                    pe_model_install_error)
from settings import MODEL_ID, MODEL_REVISION, PE_T2I_ID, PE_T2I_REVISION, ROOT, prepare_output_directory


class SetupTests(unittest.TestCase):
    def test_download_requires_explicit_license_acceptance(self):
        self.assertEqual(download(False), 2)

    def test_prompt_enhancer_download_requires_explicit_license_acceptance(self):
        self.assertEqual(download_enhancers(False), 2)

    def test_prompt_enhancer_requires_files_and_identity_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error = pe_model_install_error(root, PE_T2I_ID, PE_T2I_REVISION)
            self.assertIn("Incomplete", error)

    def test_missing_model_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = missing_model_files(Path(directory))
            self.assertIn("model_index.json", missing)
            self.assertIn("LICENSE", missing)

    def test_shard_presence_and_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "text_encoder"
            encoder.mkdir()
            index = encoder / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            self.assertIn("text_encoder/missing.safetensors", missing_model_files(root))
            index.write_text("{broken")
            self.assertIn("text_encoder/model.safetensors.index.json (invalid)", missing_model_files(root))

    def test_zero_byte_required_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_index.json").touch()
            self.assertIn("model_index.json", missing_model_files(root))

    def test_hugging_face_metadata_identifies_legacy_download(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / ".cache/huggingface/download/model_index.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(MODEL_REVISION + "\netag\ntimestamp\n")
            self.assertEqual(installed_model_revision(Path(directory)), MODEL_REVISION)

    def test_legacy_runtime_marker_is_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / LEGACY_MODEL_MARKER
            legacy.write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION}))
            self.assertTrue(migrate_legacy_model_marker(root))
            self.assertFalse(legacy.exists())
            self.assertTrue((root / MODEL_MARKER).is_file())
            self.assertEqual(installed_model_revision(root), MODEL_REVISION)

    def test_custom_output_directory_must_be_dedicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unrelated.txt").write_text("private")
            with self.assertRaises(RuntimeError):
                prepare_output_directory(root)

    def test_output_directory_cannot_contain_application(self):
        with self.assertRaises(RuntimeError):
            prepare_output_directory(ROOT.parent)


if __name__ == "__main__":
    unittest.main()

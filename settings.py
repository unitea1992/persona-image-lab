"""Paths and pinned model identity shared by setup, inference, and benchmarks."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_VERSION = (ROOT / "VERSION").read_text().strip()
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
MODEL_DIR = Path(os.environ.get("PERSONA_MODEL_DIR",
                                os.environ.get("SPARK_MODEL_DIR", ROOT / "model"))).resolve()
DEFAULT_OUTPUTS = (ROOT / "outputs").resolve()
OUTPUTS = Path(os.environ.get("PERSONA_OUTPUT_DIR",
                              os.environ.get("SPARK_OUTPUT_DIR", DEFAULT_OUTPUTS))).resolve()
PERSONA_DIR = Path(os.environ.get("PERSONA_DIR", ROOT / "data" / "personas")).resolve()


def prepare_output_directory(path=OUTPUTS):
    path = Path(path).resolve()
    if ROOT.is_relative_to(path) or MODEL_DIR.is_relative_to(path):
        raise RuntimeError("PERSONA_OUTPUT_DIR must not contain the application or model directory.")
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".persona-image-lab-output"
    legacy_marker = path / ".spark-image-lab-output"
    if not marker.is_file() and not legacy_marker.is_file():
        if path != DEFAULT_OUTPUTS and any(path.iterdir()):
            raise RuntimeError("A custom PERSONA_OUTPUT_DIR must be empty or already initialized by Persona Image Lab.")
        marker.write_text("Persona Image Lab output directory\n")
    return path

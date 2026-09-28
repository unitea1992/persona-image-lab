"""Paths and pinned model identity shared by setup, inference, and benchmarks."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_VERSION = (ROOT / "VERSION").read_text().strip()
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
MODEL_DIR = Path(os.environ.get("PERSONA_MODEL_DIR", ROOT / "model")).resolve()
PE_T2I_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_T2I_REVISION = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"
PE_I2I_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
PE_I2I_REVISION = "72927bc08afc99b7888ceb7d7d51a12db3700bbd"
PE_T2I_DIR = Path(os.environ.get("PERSONA_PE_T2I_DIR", MODEL_DIR / "prompt-enhancer-t2i")).resolve()
PE_I2I_DIR = Path(os.environ.get("PERSONA_PE_I2I_DIR", MODEL_DIR / "prompt-enhancer-i2i")).resolve()
PROMPT_ENHANCER_ENABLED = os.environ.get("PERSONA_PROMPT_ENHANCER", "1") != "0"
PROMPT_ENHANCER_BACKEND = os.environ.get("PERSONA_PE_BACKEND", "auto").strip().lower()
TORCH_COMPILE_ENABLED = os.environ.get("PERSONA_TORCH_COMPILE", "1") != "0"
FP8_MODE = os.environ.get("PERSONA_FP8", "off").strip().lower()
try:
    DEFAULT_STEPS = int(os.environ.get("PERSONA_DEFAULT_STEPS", "20"))
except ValueError:
    DEFAULT_STEPS = 20
if DEFAULT_STEPS not in range(1, 81):
    DEFAULT_STEPS = 20
PE_T2I_SOCKET = Path(os.environ.get(
    "PERSONA_PE_T2I_SOCKET", ROOT / "cache" / "prompt-enhancer" / "t2i.sock"
)).resolve()
PE_I2I_SOCKET = Path(os.environ.get(
    "PERSONA_PE_I2I_SOCKET", ROOT / "cache" / "prompt-enhancer" / "i2i.sock"
)).resolve()
DEFAULT_OUTPUTS = (ROOT / "outputs").resolve()
OUTPUTS = Path(os.environ.get("PERSONA_OUTPUT_DIR", DEFAULT_OUTPUTS)).resolve()
PERSONA_DIR = Path(os.environ.get("PERSONA_DIR", ROOT / "data" / "personas")).resolve()


def prepare_output_directory(path=OUTPUTS):
    path = Path(path).resolve()
    if ROOT.is_relative_to(path) or MODEL_DIR.is_relative_to(path):
        raise RuntimeError("PERSONA_OUTPUT_DIR must not contain the application or model directory.")
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".persona-image-lab-output"
    if not marker.is_file():
        if path != DEFAULT_OUTPUTS and any(path.iterdir()):
            raise RuntimeError("A custom PERSONA_OUTPUT_DIR must be empty or already initialized by Persona Image Lab.")
        marker.write_text("Persona Image Lab output directory\n")
    return path

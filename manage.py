"""Preflight and explicit, revision-pinned model download."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from settings import (MODEL_DIR, MODEL_ID, MODEL_REVISION, OUTPUTS,
                      PE_I2I_DIR, PE_I2I_ID, PE_I2I_REVISION,
                      PE_T2I_DIR, PE_T2I_ID, PE_T2I_REVISION,
                      PROMPT_ENHANCER_ENABLED, prepare_output_directory)

REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
MODEL_MARKER = ".persona-model.json"
LEGACY_MODEL_MARKER = "." + "spark" + "-model.json"
PE_REQUIRED_FILES = [
    "LICENSE", "chat_template.jinja", "config.json", "generation_config.json",
    "model.safetensors.index.json", "processor_config.json", "system_prompt.txt",
    "tokenizer.json", "tokenizer_config.json",
]


def missing_model_files(directory):
    required = ["model_index.json", "LICENSE", "scheduler/scheduler_config.json",
                "processor/tokenizer.json", "processor/tokenizer_config.json",
                "processor/preprocessor_config.json", "processor/chat_template.jinja",
                "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
                "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    missing = [name for name in required
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    for name in required:
        if not name.endswith(".index.json") or name in missing:
            continue
        try:
            index = json.loads((directory / name).read_text())
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("invalid weight map")
            shards = set(weight_map.values())
            for shard in shards:
                relative = Path(name).parent / shard
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(relative))
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(f"{name} (invalid)")
    return missing


def _marker_revision(path):
    try:
        identity = json.loads(path.read_text())
        if identity.get("model") == MODEL_ID and REVISION_PATTERN.fullmatch(identity.get("revision", "")):
            return identity["revision"]
    except (AttributeError, json.JSONDecodeError, OSError):
        return None
    return None


def migrate_legacy_model_marker(directory):
    directory = Path(directory)
    marker = directory / MODEL_MARKER
    legacy = directory / LEGACY_MODEL_MARKER
    current = _marker_revision(marker) if marker.is_file() else None
    legacy_revision = _marker_revision(legacy) if legacy.is_file() else None
    if current:
        if legacy.is_file():
            legacy.unlink()
        return False
    if not legacy_revision:
        return False
    marker.write_text(json.dumps({"model": MODEL_ID, "revision": legacy_revision}, indent=2) + "\n")
    legacy.unlink()
    return True


def installed_model_revision(directory):
    directory = Path(directory)
    marker = directory / MODEL_MARKER
    if marker.is_file():
        return _marker_revision(marker)
    legacy = directory / LEGACY_MODEL_MARKER
    if legacy.is_file():
        revision = _marker_revision(legacy)
        if revision:
            return revision
    metadata = directory / ".cache/huggingface/download/model_index.json.metadata"
    try:
        revision = metadata.read_text().splitlines()[0].strip()
    except (IndexError, OSError):
        return None
    return revision if REVISION_PATTERN.fullmatch(revision) else None


def pe_model_install_error(directory, model_id, revision):
    directory = Path(directory)
    missing = [name for name in PE_REQUIRED_FILES
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and index_path.stat().st_size:
        try:
            index = json.loads(index_path.read_text())
            for shard in set(index.get("weight_map", {}).values()):
                if not shard or not (directory / shard).is_file() or (directory / shard).stat().st_size == 0:
                    missing.append(str(shard or "invalid shard"))
        except (AttributeError, json.JSONDecodeError, OSError):
            missing.append("model.safetensors.index.json (invalid)")
    if missing:
        return f"Incomplete {model_id}. Missing or invalid: {', '.join(missing)}"
    marker = directory / ".persona-pe-model.json"
    try:
        identity = json.loads(marker.read_text())
    except (AttributeError, json.JSONDecodeError, OSError):
        identity = {}
    if identity.get("model") != model_id or identity.get("revision") != revision:
        return f"{model_id} revision is unverified. Re-run ./persona download-enhancers --accept-model-license."
    return None


def model_install_error(directory):
    missing = missing_model_files(directory)
    if missing:
        return f"Incomplete model. Missing or invalid: {', '.join(missing)}"
    revision = installed_model_revision(directory)
    if revision != MODEL_REVISION:
        found = revision or "unverified"
        return f"Model revision mismatch: expected {MODEL_REVISION}, found {found}. Re-run the download command."
    return None


def doctor(require_model=False):
    import torch
    from persona_profiles import load_personas
    from settings import PERSONA_DIR

    failures = []
    print(f"Architecture: {platform.machine()}")
    if platform.machine() not in ("aarch64", "arm64"):
        failures.append("This build targets ARM64 DGX Spark / GB10 systems.")
    if not torch.cuda.is_available():
        failures.append("CUDA unavailable. Check the NVIDIA driver, Container Toolkit, and GPU access.")
    else:
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}; memory: {gpu.total_memory / 2**30:.1f} GiB")
        if "GB10" not in gpu.name:
            failures.append("This release is validated for GB10, not this GPU.")
    try:
        prepare_output_directory()
        if not os.access(OUTPUTS, os.W_OK):
            failures.append("Output directory is not writable; check LOCAL_UID and LOCAL_GID.")
        print(f"Output disk free: {shutil.disk_usage(OUTPUTS).free / 2**30:.1f} GiB")
    except (OSError, RuntimeError) as error:
        failures.append(str(error))
    if migrate_legacy_model_marker(MODEL_DIR):
        print("Model marker: migrated to Persona Image Lab naming")
    model_error = model_install_error(MODEL_DIR)
    print("Model: " + (model_error or "required files and revision verified"))
    enhancer_errors = []
    if PROMPT_ENHANCER_ENABLED:
        for label, directory, model_id, revision in (
            ("PE-T2I", PE_T2I_DIR, PE_T2I_ID, PE_T2I_REVISION),
            ("PE-I2I", PE_I2I_DIR, PE_I2I_ID, PE_I2I_REVISION),
        ):
            error = pe_model_install_error(directory, model_id, revision)
            enhancer_errors.append(error)
            print(f"{label}: " + (error or "required files and revision verified"))
    personas = load_personas()
    print(f"Persona presets: {len(personas)} ({PERSONA_DIR})")
    if require_model and model_error:
        failures.append(model_error + " Run ./persona download --accept-model-license.")
    if require_model and PROMPT_ENHANCER_ENABLED:
        failures.extend(error + " Run ./persona download-enhancers --accept-model-license."
                        for error in enhancer_errors if error)
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


def download(accepted):
    if not accepted:
        print("Read Qwen's model license first:\n"
              f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE\n"
              "It limits use to research/evaluation; commercial use needs a separate license.\n"
              "Re-run with --accept-model-license only if you accept those terms.", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if missing_model_files(MODEL_DIR) and shutil.disk_usage(MODEL_DIR).free < 40 * 2**30:
        print("Allow at least 40 GiB free for the model download and temporary files.", file=sys.stderr)
        return 1
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL_DIR))
    missing = missing_model_files(MODEL_DIR)
    if missing:
        print(f"Incomplete download: {', '.join(missing)}", file=sys.stderr)
        return 1
    (MODEL_DIR / MODEL_MARKER).write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n")
    legacy_marker = MODEL_DIR / LEGACY_MODEL_MARKER
    if legacy_marker.is_file():
        legacy_marker.unlink()
    print(f"Model ready: {MODEL_DIR}")
    return 0


def download_enhancers(accepted):
    if not accepted:
        print("Read the Qwen Research License first. The prompt-enhancer models are "
              "licensed for non-commercial research/evaluation unless you obtain a separate commercial license.\n"
              f"https://huggingface.co/{PE_T2I_ID}/blob/{PE_T2I_REVISION}/LICENSE\n"
              "Re-run with --accept-model-license only if you accept those terms.", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    targets = (
        (PE_T2I_ID, PE_T2I_REVISION, PE_T2I_DIR),
        (PE_I2I_ID, PE_I2I_REVISION, PE_I2I_DIR),
    )
    for model_id, revision, directory in targets:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_id} @ {revision} ...", flush=True)
        snapshot_download(repo_id=model_id, revision=revision, local_dir=str(directory))
        error = pe_model_install_error(directory, model_id, revision)
        if error and "revision is unverified" not in error:
            print(error, file=sys.stderr)
            return 1
        (directory / ".persona-pe-model.json").write_text(
            json.dumps({"model": model_id, "revision": revision}, indent=2) + "\n"
        )
        error = pe_model_install_error(directory, model_id, revision)
        if error:
            print(error, file=sys.stderr)
            return 1
        print(f"Prompt enhancer ready: {directory}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor")
    check.add_argument("--require-model", action="store_true")
    fetch = sub.add_parser("download")
    fetch.add_argument("--accept-model-license", action="store_true")
    fetch_pe = sub.add_parser("download-enhancers")
    fetch_pe.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        result = doctor(args.require_model)
    elif args.command == "download-enhancers":
        result = download_enhancers(args.accept_model_license)
    else:
        result = download(args.accept_model_license)
    sys.exit(result)

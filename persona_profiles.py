"""Load private, host-mounted persona presets without copying them into Git."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from settings import PERSONA_DIR

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_SUFFIXES = {".md", ".txt"}
MAX_CONFIG_BYTES = 64 * 1024
MAX_PROMPT_BYTES = 128 * 1024
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonaProfile:
    identifier: str
    name: str
    description: str
    prompt_prefix: str
    references: tuple[Path, ...]
    reference_labels: tuple[str, ...]


def _safe_child(root, relative, suffixes):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("Persona file paths must be non-empty strings.")
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError("Persona file paths must be relative.")
    candidate = root / raw
    resolved = candidate.resolve()
    if candidate.is_symlink() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("Persona file paths must stay inside the persona directory.")
    if not candidate.is_file() or candidate.suffix.lower() not in suffixes:
        raise ValueError(f"Persona file is missing or unsupported: {relative}")
    return candidate


def _load_profile(directory):
    config_path = directory / "persona.json"
    if not config_path.is_file() or config_path.is_symlink():
        return None
    if config_path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"{directory.name}/persona.json is too large.")
    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{directory.name}/persona.json is invalid.") from error
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{directory.name}/persona.json must use schema_version 1.")

    name = data.get("name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
        raise ValueError(f"{directory.name}: name must be 1-80 characters.")
    description = data.get("description", "")
    if not isinstance(description, str) or len(description) > 240:
        raise ValueError(f"{directory.name}: description must be at most 240 characters.")

    prompt_parts = []
    prompt_prefix = data.get("prompt_prefix", "")
    if not isinstance(prompt_prefix, str):
        raise ValueError(f"{directory.name}: prompt_prefix must be text.")
    if prompt_prefix.strip():
        prompt_parts.append(prompt_prefix.strip())
    prompt_file = data.get("prompt_file")
    if prompt_file is not None:
        prompt_path = _safe_child(directory, prompt_file, TEXT_SUFFIXES)
        if prompt_path.stat().st_size > MAX_PROMPT_BYTES:
            raise ValueError(f"{directory.name}: prompt file is too large.")
        prompt_parts.append(prompt_path.read_text().strip())

    references = data.get("references", [])
    if not isinstance(references, list) or len(references) > 10:
        raise ValueError(f"{directory.name}: references must contain at most 10 images.")
    reference_paths = []
    labels = []
    seen = set()
    for index, reference in enumerate(references, start=1):
        if not isinstance(reference, dict):
            raise ValueError(f"{directory.name}: reference {index} must be an object.")
        path = _safe_child(directory, reference.get("path"), IMAGE_SUFFIXES)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"{directory.name}: duplicate reference image.")
        seen.add(resolved)
        label = reference.get("label", f"Reference {index}")
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 80:
            raise ValueError(f"{directory.name}: reference labels must be 1-80 characters.")
        reference_paths.append(path)
        labels.append(label.strip())

    return PersonaProfile(
        identifier=directory.name,
        name=name.strip(),
        description=description.strip(),
        prompt_prefix="\n\n".join(part for part in prompt_parts if part),
        references=tuple(reference_paths),
        reference_labels=tuple(labels),
    )


def load_personas(root=PERSONA_DIR):
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    profiles = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            profile = _load_profile(directory)
        except (ValueError, OSError, UnicodeError) as error:
            LOG.warning("Skipping invalid persona preset %s: %s", directory.name, error)
            continue
        if profile is not None:
            profiles.append(profile)
    return profiles


def get_persona(identifier, root=PERSONA_DIR):
    if not identifier:
        return None
    if not isinstance(identifier, str) or "/" in identifier or "\\" in identifier:
        raise ValueError("Invalid persona selection.")
    root = Path(root).resolve()
    directory = root / identifier
    resolved = directory.resolve()
    if directory.is_symlink() or not resolved.is_relative_to(root) or not directory.is_dir():
        raise ValueError("Persona preset is unavailable.")
    try:
        profile = _load_profile(directory)
    except (OSError, UnicodeError) as error:
        raise ValueError("Persona preset is unavailable.") from error
    if profile is None:
        raise ValueError("Persona preset is unavailable.")
    return profile


def compose_prompt(profile, prompt):
    prompt = prompt.strip()
    if profile is None or not profile.prompt_prefix:
        return prompt
    return f"{profile.prompt_prefix}\n\n{prompt}"

# Persona Image Lab

A small, local-first character image studio for **NVIDIA DGX Spark**, powered by
Qwen-Image-2.1 and Gradio.

Persona Image Lab is a public fork of
[Spark Image Lab](https://github.com/joeynyc/spark-image-lab). It keeps the
upstream project's DGX Spark-focused Docker setup and generation history, while
adding private character presets and a simpler character-first workflow.

## Why this exists

The application repository is safe to keep public. Character definitions,
identity images, prompt canon, model weights, generations, and local settings
stay outside Git.

The everyday flow is intentionally small:

1. Choose a private character preset.
2. Describe the new scene.
3. Optionally add outfit, pose, or style references.
4. Choose a canvas preset and generate.

Advanced width, height, step, and seed controls remain available without
occupying the main interface.

## Requirements

- NVIDIA DGX Spark / GB10
- Docker with NVIDIA GPU support
- Docker Compose 2.30+
- About 80 GiB free disk space for the container, model, build cache, and output

The host does not need a Python, PyTorch, Diffusers, or Gradio environment.
Application code is copied into the local Docker image at build time rather
than bind-mounted from the checkout.

## Quick start

```bash
git clone https://github.com/unitea1992/persona-image-lab.git
cd persona-image-lab
./persona build
./persona doctor
./persona download --accept-model-license
./persona start
```

Open <http://127.0.0.1:7862/> on the Spark. From another computer, keep the
application bound to loopback and use an SSH tunnel:

```bash
ssh -N -L 7862:127.0.0.1:7862 YOUR_SPARK_SSH_ALIAS
```

Useful commands:

```bash
./persona status
./persona logs
./persona test
./persona stop
```

## Private persona presets

By default, persona data is read from `./data/personas`, which is ignored by
Git. For real character data, keeping the files outside the repository is
recommended:

```bash
cp .env.example .env
```

Set the host directory in `.env`:

```dotenv
PERSONA_DATA_DIR=/absolute/path/to/private/personas
```

That directory is mounted **read-only** at runtime. The launcher rejects a
persona directory that overlaps `model/`, `outputs/`, or `cache/`, including a
parent/child overlap that would make part of the private tree writable through
another mount.

Each persona is one directory containing `persona.json` and its private source
files:

```text
private-personas/
└── sample-character/
    ├── persona.json
    ├── visual-canon.md
    ├── identity.png
    └── fullbody.png
```

Example `persona.json`:

```json
{
  "schema_version": 1,
  "name": "Sample Character",
  "description": "Private local character preset",
  "prompt_file": "visual-canon.md",
  "references": [
    {"label": "Identity", "path": "identity.png"},
    {"label": "Full body", "path": "fullbody.png"}
  ]
}
```

`prompt_prefix` may be used instead of, or together with, `prompt_file`.
Persona references and manually uploaded references may total at most ten
images, matching the current Qwen-Image-2.1 workflow.

Persona file paths must remain inside their own preset directory. Symlinked
persona directories and symlinked referenced files are rejected.

## Data boundaries

The following are intentionally excluded from Git:

- `data/` — optional local persona data
- `model/` — downloaded model weights
- `outputs/` — generated PNGs, metadata, and retained reference copies
- `cache/` — package/model caches
- `.env` — machine-specific configuration

The application checkout itself is not mounted into the running container.
Only `model/`, `outputs/`, and `cache/` are writable bind mounts. Persona data
is mounted separately as read-only, including when the ignored
`./data/personas` default is used.

The supported Compose configuration binds the browser port to
`127.0.0.1`. Public Gradio sharing and Hugging Face/Gradio telemetry are
disabled. This is a trusted single-owner tool, not a public multi-user service.

## Current model

The first engine is the pinned Qwen-Image-2.1 Diffusers pipeline used by the
upstream project. The code intentionally does not add a model abstraction layer
until a second engine is actually needed.

Generation history remains compatible with the upstream JSON records. When a
private persona is used, its reference files are copied into the local output
history just like manually supplied references, so an old generation can still
be restored without the persona preset remaining installed.

## Upstream and attribution

Persona Image Lab is derived from
[joeynyc/spark-image-lab](https://github.com/joeynyc/spark-image-lab). The
upstream MIT license and copyright notice are preserved in this repository.
See [NOTICE](NOTICE) for model and third-party notices.

The application code is MIT licensed. Model weights are downloaded separately
and remain subject to their own terms.

# Persona Image Lab

[日本語](README.ja.md)

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
4. Let the prompt enhancer choose a canvas, or pick one yourself, and generate.

Short natural-language prompts are expanded locally by the official Qwen
Image 2.1 prompt-enhancer family. Text-only requests use PE-T2I; persona or
reference-image requests use PE-I2I. Advanced width, height, step, and seed controls remain available without
occupying the main interface.

## Requirements

- NVIDIA DGX Spark / GB10
- Docker with NVIDIA GPU support
- Docker Compose 2.30+
- Host vLLM plus `ninja` is recommended for accelerated PE-T2I / PE-I2I. The
  validated GB10 setup uses vLLM 0.27.1; a slower in-container Transformers
  fallback remains available.
- About 140 GiB free disk space for the container, image model, both prompt
  enhancers, build cache, and output

The image-generation application does not depend on a host PyTorch, Diffusers, or
Gradio environment. Application code is copied into the local Docker image rather
than bind-mounted from the checkout. The optional accelerated prompt-enhancer path
uses the host vLLM installation.

## Quick start

```bash
git clone https://github.com/unitea1992/persona-image-lab.git
cd persona-image-lab
./persona build
./persona doctor
./persona download --accept-model-license
./persona download-enhancers --accept-model-license
./persona start
```

`./persona start` waits for the image service and the required prompt-enhancer
services to become ready while printing their startup state. The default wait
limit is 600 seconds; override it for a slow first boot with
`PERSONA_START_TIMEOUT=900 ./persona start`.

Open <http://127.0.0.1:7862/> on the Spark. For another computer on the same
Tailscale tailnet, keep the application bound to loopback and expose it with
Tailscale Serve:

```bash
tailscale serve --bg 7862
```

Then open the HTTPS URL shown by `tailscale serve status`. Tailscale Serve can
stay configured while Persona Image Lab is stopped; only start the application
when image generation is needed.

An SSH tunnel remains a useful fallback:

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

The two prompt-enhancer checkpoints are separate Qwen3.5-VL 9B models. When host
vLLM is available, `./persona start` loads both as FP8 services and exposes them
only through Unix-domain sockets under `cache/prompt-enhancer/`. `./persona stop`
terminates them together with the image-generation service. Prompt enhancement
is enabled by default. Intermediate image previews are opt-in from Advanced
settings because raw denoising frames remain visibly noisy.

Generation progress is also shown persistently below the result instead of only
in Gradio's transient progress overlay. History selection is ignored while a
generation is active so a live preview cannot overwrite a history view. The
recent gallery has a compact selection mode that puts checkbox-style markers on
the thumbnails for confirmed batch deletion. Intermediate image previews are
disabled by default because raw denoising frames remain visibly noisy; users can
opt in from Advanced settings, where previews are limited to the final denoising stages.

An active UI generation can be stopped. Prompt-enhancer streaming is closed when
stopped during prompt expansion, and denoising is interrupted at the next step
when image generation is already running. Cancelled generations are not saved to
history.

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
  "prompt_prefix": "Reference image 1 defines this character's identity. Keep the same person and follow the user's scene request.",
  "prompt_file": "visual-canon.md",
  "references": [
    {"label": "Identity", "path": "identity.png"},
    {"label": "Full body", "path": "fullbody.png"}
  ]
}
```

`prompt_prefix` is the concise instruction sent to Qwen-Image together with the
user's prompt. Keep it short and describe the role of the reference images rather
than pasting a character specification sheet into the model prompt.

`prompt_file` is optional private canon/documentation. It is validated and kept
with the preset, but its Markdown contents are **not** concatenated into the model
prompt. This prevents long character sheets from being interpreted as visible
text or a reference-sheet layout.

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
The fork relationship is retained for attribution and change discovery, but
upstream changes are reviewed and selectively ported rather than merged or
synced automatically. See [upstream policy](docs/upstream.md) and [NOTICE](NOTICE).

The application code is MIT licensed. Model weights are downloaded separately
and remain subject to their own terms.

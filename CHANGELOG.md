# Changelog

## 0.2.0-alpha.1

- Forked the DGX Spark-focused runtime as Persona Image Lab.
- Added private, read-only persona presets with optional prompt canon and up to
  ten combined preset/manual reference images.
- Simplified the default UI around character preset, prompt, reference images,
  canvas preset, and generation; low-level size/step/seed controls moved under
  Advanced.
- Added `./persona` as the Docker helper for build, setup, run, logs, and tests.
- Renamed public Docker/runtime configuration to Persona Image Lab and documented
  the boundary between public application code and private character data.
- Removed upstream runtime compatibility names after validating the fork on a
  GB10 system with the pinned Qwen-Image-2.1 model and persona-reference path.

## Unreleased

- Add Qwen-Image-2.1 PE-T2I / PE-I2I routing for short natural-language prompts.
- Run the prompt enhancers through host vLLM with FP8 on GB10 when available,
  communicate over private Unix sockets, and stop them with the Lab service.
- Restrict manual reference paths to Gradio uploads or saved generations, lock
  prompt-enhancer runtime files to the local user, and migrate the legacy model
  identity marker to Persona Image Lab naming.
- Use prompt-enhancer aspect-ratio output for the new automatic canvas mode.
- Make the Gradio studio full-width, use the built-in Ocean theme, and show
  intermediate generation previews by default.
- Add confirmed permanent deletion for saved generations and clean up reference
  copies only after their final use.
- Document tailnet-only Tailscale Serve as the preferred remote-access path while
  keeping the image-generation container on-demand rather than always running.
- Add Japanese operator documentation for daily use, private personas,
  troubleshooting, and selective upstream updates.
- Stop injecting full persona canon Markdown into Qwen-Image prompts; keep a
  concise reference-role instruction and restore only the original user prompt.
- Simplify and localize the Gradio studio UI in Japanese, surface deletion next
  to the selected result, and report denoising step progress with an ETA.
- Keep generation progress visible below the result, prevent history selection
  from racing with an active generation, and add confirmed multi-select history
  deletion.
- Make `./persona start` wait for Image / PE-T2I / PE-I2I readiness while
  printing startup progress instead of requiring log monitoring.
- Pin CI to Ubuntu 24.04 and update checkout/setup-python to their current
  Node 24-based major versions before the runner migrations become mandatory.

## 0.1.0-alpha.2 - 2026-09-20

- Prepare the project for its public alpha release with private vulnerability
  reporting guidance and patched versions of vulnerable packages inherited from
  the NVIDIA base image.
- Run the real Gradio callback integration tests in CPU-only CI.
- Verify the installed model revision before inference and contain saved reference
  files even when the reference directory has been replaced by a symlink.

## 0.1.0-alpha.1 - 2026-09-20

Initial private DGX Spark-focused repository, validated first on ASUS GX10/GB10.

- Replace fixed demo rows with persistent generation history and a live-updating gallery.
- Restore prompt, references, dimensions, steps, seed, output and timing from history.
- Retain reference uploads locally and read existing prototype PNG/JSON records.
- Preserve the familiar Gradio layout; remove default footer links.
- Add revision-pinned model setup, hardware preflight, Compose, and launcher commands.
- Add CPU tests, container integration tests, CI, contribution and security guidance.
- Document measured batch performance, licensing boundaries, and public-release gates.

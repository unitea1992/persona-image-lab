import argparse
import importlib.metadata
import json
import logging
import os
import secrets
import threading
import time

from PIL import Image, ImageOps

try:
    from torch import OutOfMemoryError as TorchOutOfMemoryError
except ImportError:
    class TorchOutOfMemoryError(RuntimeError):
        """Fallback used by CPU-only callback tests."""

from history import (delete_generation, load_history, restore_generation,
                     save_generation, validate_request)
from manage import model_install_error
from persona_profiles import compose_prompt, get_persona, load_personas
from settings import (APP_VERSION, MODEL_DIR, MODEL_ID, MODEL_REVISION, OUTPUTS,
                      ROOT, prepare_output_directory)

DEMOS = json.loads((ROOT / "demos.json").read_text())
LOCK = threading.Lock()
HISTORY_LOCK = threading.Lock()
PIPE = None
APP_CSS = """
.gradio-container { max-width: 1440px !important; }
#studio-header { padding: 0.25rem 0 0.75rem; }
#studio-header h1 { margin-bottom: 0.2rem; letter-spacing: -0.025em; }
#studio-header p { margin: 0; opacity: 0.72; }
#result-panel img { object-fit: contain !important; }
@media (max-width: 640px) {
  #generation-history table { min-width: 800px; }
  #generation-history td:first-child { min-width: 260px; }
}
"""


def pipeline():
    global PIPE
    if PIPE is None:
        import torch
        from diffusers import QwenImage21Pipeline

        model_error = model_install_error(MODEL_DIR)
        if model_error:
            raise RuntimeError(model_error)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Start the container with GPU access; see docs/troubleshooting.md.")
        started = time.perf_counter()
        print("Loading Qwen Image 2.1 in BF16 on CUDA", flush=True)
        PIPE = QwenImage21Pipeline.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        print(f"Model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    return PIPE


def generate(prompt, references=None, width=1024, height=1024, steps=40, seed=42, context=None):
    import torch

    prepare_output_directory()
    prompt, width, height, steps, seed = validate_request(prompt, width, height, steps, seed)
    references = references or []
    if len(references) > 10:
        raise ValueError("Use at most 10 reference images.")
    images = []
    for reference in references:
        with Image.open(reference) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGBA"))
    with LOCK, torch.inference_mode():
        pipe = pipeline()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            image = pipe(
                prompt=prompt, image=images or None, width=width, height=height,
                num_inference_steps=steps, true_cfg_scale=1.0, use_kv_cache=True,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
        except Exception:
            torch.cuda.empty_cache()
            raise
        elapsed = time.perf_counter() - started
        memory_gib = torch.cuda.max_memory_allocated() / 2**30
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    alpha_extrema = alpha.getextrema() if alpha is not None else None
    metadata = {
        "app_version": APP_VERSION,
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "prompt": prompt,
        "width": image.width, "height": image.height, "steps": steps,
        "seed": seed, "true_cfg_scale": 1.0, "use_kv_cache": True,
        "elapsed_seconds": round(elapsed, 2), "peak_allocated_gib": round(memory_gib, 2),
        "mode": image.mode, "alpha_extrema": alpha_extrema,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "diffusers", "transformers")},
    }
    if context:
        metadata["context"] = context
    with HISTORY_LOCK:
        result = save_generation(OUTPUTS, image, metadata, references)
    print(f"Saved {result[2]['id']} in {elapsed:.1f}s", flush=True)
    return result


def batch(limit):
    results = {}
    for demo in DEMOS[:limit]:
        reference = demo.get("reference")
        image, _, _ = generate(
            demo["prompt"], [results[reference]] if reference else None,
            demo["width"], demo["height"], 40, demo["seed"],
        )
        results[demo["name"]] = image
        manifest = OUTPUTS / "demo-manifest.json"
        manifest.write_text(json.dumps(results, indent=2) + "\n")


def build_app():
    import gradio as gr

    size_presets = {
        "Square · 1024 × 1024": (1024, 1024),
        "Portrait · 832 × 1216": (832, 1216),
        "Landscape · 1216 × 832": (1216, 832),
        "Tall · 768 × 1344": (768, 1344),
        "Custom": None,
    }
    personas = load_personas()
    persona_choices = [("No character preset", "")] + [
        (f"{profile.name} · {profile.identifier}", profile.identifier)
        for profile in personas
    ]

    def stats_text(stats):
        context = stats.get("context")
        persona_name = context.get("persona_name") if isinstance(context, dict) else None
        prefix = f"{persona_name} | " if persona_name else ""
        return f"{prefix}{stats.get('elapsed_seconds', 0):.1f}s | {stats['width']} × {stats['height']} | Seed {stats['seed']} | Peak {stats.get('peak_allocated_gib', 0):.1f} GiB"

    def refresh():
        entries = load_history(OUTPUTS)
        samples = [[r['prompt'], ", ".join(ref.get('filename', 'Reference')
                                         for ref in r.get('references', [])),
                    r['width'], r['height'],
                    r['steps'], r['seed'], r.get('elapsed_seconds', 0)] for r in entries]
        return gr.Dataset(samples=samples), [r['image_path'] for r in entries], [r['id'] for r in entries]

    def persona_info(identifier):
        if not identifier:
            return "No private character preset selected. Extra reference images are still available below."
        try:
            profile = get_persona(identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        labels = ", ".join(profile.reference_labels) if profile.reference_labels else "no saved references"
        description = profile.description or "Private local preset"
        return f"**{profile.name}** — {description}  \nReferences: {labels}"

    def run(persona_id, prompt, refs, size_preset, width, height, steps, seed, randomize_seed):
        try:
            profile = get_persona(persona_id)
            if size_preset not in size_presets:
                raise ValueError("Choose an available canvas preset.")
            dimensions = size_presets[size_preset]
            if dimensions is not None:
                width, height = dimensions
            if randomize_seed:
                seed = secrets.randbelow(2**32)
            persona_refs = [str(path) for path in profile.references] if profile else []
            references = persona_refs + (refs or [])
            if len(references) > 10:
                raise ValueError("Persona and extra references may total at most 10 images.")
            effective_prompt = compose_prompt(profile, prompt)
            context = ({"persona_id": profile.identifier, "persona_name": profile.name}
                       if profile else None)
            image, metadata, stats = generate(
                effective_prompt, references, width, height, steps, seed, context=context,
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except TorchOutOfMemoryError as error:
            raise gr.Error("GPU memory exhausted. Reduce the image dimensions or reference count.") from error
        except (OSError, Image.DecompressionBombError) as error:
            logging.exception("Image input or output failed")
            raise gr.Error("Could not read an image or save the result. Check the image files, free disk space, and output permissions.") from error
        return image, [image, metadata], stats_text(stats), *refresh()

    def restore(index, identifiers):
        if not isinstance(index, int) or not 0 <= index < len(identifiers):
            raise gr.Error("Select an available generation.")
        try:
            entry = restore_generation(OUTPUTS, identifiers[index])
        except (ValueError, OSError) as error:
            raise gr.Error("This generation is unavailable. Refresh the page to reload history.") from error
        if entry['missing_references']:
            gr.Warning("Some original reference images are unavailable. Re-upload them before regenerating this edit.")
        return ("", entry['prompt'], entry['reference_paths'], "Custom",
                entry['width'], entry['height'], entry['steps'], entry['seed'], False,
                entry['image_path'],
                [entry['image_path'], entry['metadata_path']], stats_text(entry),
                entry['id'], False)

    def select_image(identifiers, event: gr.SelectData):
        return restore(event.index, identifiers)

    def use_reference(path):
        if not path:
            raise gr.Error("Generate an image first.")
        return [path]

    def delete_selected(identifier, confirmed):
        if not identifier:
            raise gr.Error("Select a generation to delete.")
        if confirmed is not True:
            raise gr.Error("Confirm permanent deletion first.")
        try:
            with HISTORY_LOCK:
                delete_generation(OUTPUTS, identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except OSError as error:
            logging.exception("Generation deletion failed")
            raise gr.Error("Could not delete every generation file. Check output permissions.") from error
        return None, None, None, "", None, False, *refresh()

    def clear_delete_selection():
        return None, False

    with gr.Blocks(title="Persona Image Lab") as app:
        gr.HTML(
            '<div id="studio-header"><h1>Persona Image Lab</h1>'
            '<p>Character-first local image generation on DGX Spark · Qwen-Image-2.1</p></div>'
        )
        identifiers = gr.State([])
        selected_identifier = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                persona = gr.Dropdown(
                    choices=persona_choices,
                    value="",
                    label="Character preset",
                    info="Loaded from the private persona directory mounted into this container.",
                )
                persona_status = gr.Markdown(persona_info(""))
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=7,
                    placeholder="Describe the scene, outfit, pose, expression, and framing.",
                )
                refs = gr.File(
                    label="Extra references",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                size_preset = gr.Dropdown(
                    choices=list(size_presets),
                    value="Square · 1024 × 1024",
                    label="Canvas",
                )
                randomize_seed = gr.Checkbox(label="New variation each time", value=True)
                with gr.Accordion("Advanced", open=False):
                    gr.Markdown("Width and height are used only when **Canvas** is **Custom**.")
                    with gr.Row():
                        width = gr.Slider(512, 2752, value=1024, step=32, label="Width")
                        height = gr.Slider(512, 2752, value=1024, step=32, label="Height")
                    steps = gr.Slider(1, 80, value=40, step=1, label="Steps")
                    seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=4294967295)
                create = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                result = gr.Image(label="Result", type="filepath", image_mode="RGBA", format="png",
                                  interactive=False, height=620, elem_id="result-panel")
                reuse = gr.Button("Use result as extra reference")
                stats = gr.Textbox(label="Generation", interactive=False)
                with gr.Accordion("Downloads and history actions", open=False):
                    files = gr.File(label="PNG and generation record", file_count="multiple")
                    delete_confirm = gr.Checkbox(label="Confirm permanent deletion", value=False)
                    delete = gr.Button("Delete permanently", variant="stop")
        gallery = gr.Gallery(label="Recent generations", columns=4, height=320, preview=False)
        with gr.Accordion("Generation history", open=False):
            history = gr.Dataset(
                components=[prompt, gr.Textbox(render=False), width, height, steps, seed,
                            gr.Number(render=False)],
                headers=["Prompt", "Reference images", "Width", "Height", "Steps", "Seed", "Time (s)"],
                samples=[], type="index", layout="table", samples_per_page=10,
                label=None, elem_id="generation-history",
            )
        refresh_outputs = [history, gallery, identifiers]
        restore_outputs = [persona, prompt, refs, size_preset, width, height, steps, seed,
                           randomize_seed, result, files, stats,
                           selected_identifier, delete_confirm]
        app.load(refresh, outputs=refresh_outputs, queue=False, api_name=False)
        persona.change(persona_info, inputs=persona, outputs=persona_status, queue=False, api_name=False)
        generation = create.click(run, inputs=[persona, prompt, refs, size_preset, width, height,
                                              steps, seed, randomize_seed],
                                  outputs=[result, files, stats, *refresh_outputs],
                                  concurrency_limit=1, show_progress_on=[result],
                                  api_name="generate")
        generation.then(clear_delete_selection,
                        outputs=[selected_identifier, delete_confirm],
                        queue=False, api_name=False)
        history.click(restore, inputs=[history, identifiers], outputs=restore_outputs, queue=False, api_name=False)
        gallery.select(select_image, inputs=identifiers, outputs=restore_outputs, queue=False, api_name=False)
        reuse.click(use_reference, inputs=result, outputs=refs, queue=False)
        delete.click(delete_selected, inputs=[selected_identifier, delete_confirm],
                     outputs=[refs, result, files, stats, selected_identifier,
                              delete_confirm, *refresh_outputs], queue=False, api_name=False)
    return app.queue(max_size=8)


def serve():
    import gradio as gr

    prepare_output_directory()
    pipeline()
    build_app().launch(server_name=os.environ.get("PERSONA_HOST",
                                                  os.environ.get("SPARK_HOST", "127.0.0.1")),
                       server_port=int(os.environ.get("PERSONA_PORT",
                                                      os.environ.get("SPARK_PORT", "7860"))), share=False,
                       allowed_paths=[str(OUTPUTS)], footer_links=[], run_history=False,
                       max_file_size="25mb",
                       theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
                       css=APP_CSS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--limit", type=int, default=len(DEMOS))
    args = parser.parse_args()
    if not (args.serve or args.batch):
        parser.error("Choose --serve or --batch.")
    if args.batch:
        batch(args.limit)
    if args.serve:
        serve()

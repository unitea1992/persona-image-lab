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
.gradio-container { max-width: 1480px !important; }
#studio-header { padding: 0.25rem 0 1rem; }
#studio-header h1 { margin: 0 0 0.25rem; letter-spacing: -0.03em; }
#studio-header p { margin: 0; opacity: 0.7; }
#persona-status { min-height: 2.4rem; opacity: 0.82; }
#result-panel img { object-fit: contain !important; }
#generation-stats { min-height: 1.5rem; opacity: 0.72; }
#delete-confirmation { border-left: 3px solid var(--color-accent); padding-left: 0.8rem; }
#recent-generations { margin-top: 0.75rem; }
#result-panel button[aria-label="Share"],
#recent-generations button[aria-label="Share"] { display: none !important; }
#recent-generations button[aria-label="Download All"] { display: none !important; }
@media (max-width: 640px) {
  #studio-header { padding-bottom: 0.5rem; }
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


def generate(prompt, references=None, width=1024, height=1024, steps=40, seed=42,
             context=None, user_prompt=None, progress_callback=None):
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
        def on_step_end(_pipe, step_index, _timestep, callback_kwargs):
            if progress_callback is not None:
                progress_callback(step_index + 1, steps)
            return callback_kwargs

        try:
            image = pipe(
                prompt=prompt, image=images or None, width=width, height=height,
                num_inference_steps=steps, true_cfg_scale=1.0, use_kv_cache=True,
                generator=torch.Generator("cuda").manual_seed(seed),
                callback_on_step_end=on_step_end if progress_callback is not None else None,
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
    if isinstance(user_prompt, str) and user_prompt.strip():
        metadata["user_prompt"] = user_prompt.strip()
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
        "正方形 · 1024 × 1024": (1024, 1024),
        "縦長 · 832 × 1216": (832, 1216),
        "横長 · 1216 × 832": (1216, 832),
        "長い縦長 · 768 × 1344": (768, 1344),
        "カスタム": None,
    }
    personas = load_personas()
    persona_identifiers = {profile.identifier for profile in personas}
    persona_choices = [("プリセットなし", "")] + [
        (profile.name, profile.identifier)
        for profile in personas
    ]

    def display_prompt(entry):
        value = entry.get("user_prompt")
        if isinstance(value, str) and value.strip():
            return value
        context = entry.get("context")
        if isinstance(context, dict) and context.get("persona_id"):
            # Persona records created before user_prompt was introduced contain
            # the old internal prompt, sometimes including the full canon. Do
            # not put that implementation detail back into the user textbox.
            return ""
        return entry["prompt"]

    def canvas_for(entry):
        dimensions = (entry["width"], entry["height"])
        for label, candidate in size_presets.items():
            if candidate == dimensions:
                return label
        return "カスタム"

    def stats_text(stats):
        context = stats.get("context")
        persona_name = context.get("persona_name") if isinstance(context, dict) else None
        prefix = f"{persona_name} ・ " if persona_name else ""
        return (f"{prefix}{stats.get('elapsed_seconds', 0):.1f}秒 ・ "
                f"{stats['width']} × {stats['height']} ・ Seed {stats['seed']}")

    def refresh():
        entries = load_history(OUTPUTS)
        samples = [[display_prompt(r), ", ".join(ref.get('filename', 'Reference')
                                         for ref in r.get('references', [])),
                    r['width'], r['height'],
                    r['steps'], r['seed'], r.get('elapsed_seconds', 0)] for r in entries]
        return gr.Dataset(samples=samples), [r['image_path'] for r in entries], [r['id'] for r in entries]

    def persona_info(identifier):
        if not identifier:
            return "キャラクター固定なし。必要なら参照画像を追加できます。"
        try:
            profile = get_persona(identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        description = profile.description or "ローカルPersona"
        return f"**{profile.name}** — {description}（参照画像 {len(profile.references)}枚）"

    def run_with_progress(persona_id, prompt, refs, size_preset, width, height, steps, seed,
                          randomize_seed, progress=gr.Progress()):
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("プロンプトを入力してください。")
            profile = get_persona(persona_id)
            if size_preset not in size_presets:
                raise ValueError("利用できる画像サイズを選んでください。")
            dimensions = size_presets[size_preset]
            if dimensions is not None:
                width, height = dimensions
            if randomize_seed:
                seed = secrets.randbelow(2**32)
            persona_refs = [str(path) for path in profile.references] if profile else []
            references = persona_refs + (refs or [])
            if len(references) > 10:
                raise ValueError("Personaと追加の参照画像は、合計10枚までです。")
            effective_prompt = compose_prompt(profile, prompt)
            context = ({
                "persona_id": profile.identifier,
                "persona_name": profile.name,
                "persona_reference_count": len(persona_refs),
            } if profile else None)
            progress_started = time.perf_counter()
            progress(0, desc="生成を準備中…")

            def report_progress(step, total):
                elapsed = time.perf_counter() - progress_started
                if step < 2:
                    description = f"生成中 {step}/{total}"
                else:
                    remaining = (elapsed / step) * (total - step)
                    description = f"生成中 {step}/{total} ・ 残り約{remaining:.0f}秒"
                progress((step, total), desc=description)

            image, metadata, stats = generate(
                effective_prompt, references, width, height, steps, seed,
                context=context, user_prompt=prompt, progress_callback=report_progress,
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except TorchOutOfMemoryError as error:
            raise gr.Error("GPUメモリが不足しました。画像サイズか参照画像の枚数を減らしてください。") from error
        except (OSError, Image.DecompressionBombError) as error:
            logging.exception("Image input or output failed")
            raise gr.Error("画像を読み込むか、生成結果を保存できませんでした。画像ファイルと空き容量を確認してください。") from error
        return image, [image, metadata], stats_text(stats), *refresh(), stats["id"]

    def custom_dimensions_visibility(size_preset):
        return gr.Row(visible=size_preset == "カスタム")

    def run(persona_id, prompt, refs, size_preset, width, height, steps, seed, randomize_seed):
        return run_with_progress(
            persona_id, prompt, refs, size_preset, width, height, steps, seed,
            randomize_seed, progress=lambda *_args, **_kwargs: None,
        )

    def restore(index, identifiers):
        if not isinstance(index, int) or not 0 <= index < len(identifiers):
            raise gr.Error("生成履歴から画像を選んでください。")
        try:
            entry = restore_generation(OUTPUTS, identifiers[index])
        except (ValueError, OSError) as error:
            raise gr.Error("この生成結果は利用できません。ページを再読み込みしてください。") from error
        if entry['missing_references']:
            gr.Warning("元の参照画像の一部がありません。再生成する場合は必要な画像を追加してください。")
        context = entry.get("context") if isinstance(entry.get("context"), dict) else {}
        persona_id = context.get("persona_id", "")
        if persona_id not in persona_identifiers:
            persona_id = ""
        if persona_id and not entry.get("user_prompt"):
            gr.Warning("この履歴は旧形式のため、当時入力したプロンプトは復元できません。")
        persona_reference_count = context.get("persona_reference_count", 0)
        if not isinstance(persona_reference_count, int) or persona_reference_count < 0:
            persona_reference_count = 0
        if persona_id and "persona_reference_count" not in context:
            try:
                persona_reference_count = len(get_persona(persona_id).references)
            except ValueError:
                persona_reference_count = 0
        reference_paths = entry['reference_paths']
        extra_references = (reference_paths[persona_reference_count:]
                            if persona_id else reference_paths)
        return (persona_id, display_prompt(entry), extra_references, canvas_for(entry),
                entry['width'], entry['height'], entry['steps'], entry['seed'], False,
                entry['image_path'],
                [entry['image_path'], entry['metadata_path']], stats_text(entry),
                entry['id'])

    def select_image(identifiers, event: gr.SelectData):
        return restore(event.index, identifiers)

    def use_reference(path):
        if not path:
            raise gr.Error("先に画像を生成するか、履歴から画像を選んでください。")
        return [path]

    def request_delete(identifier):
        if not identifier:
            raise gr.Error("削除する生成結果を選んでください。")
        return gr.Group(visible=True)

    def hide_delete_confirmation():
        return gr.Group(visible=False)

    def delete_selected(identifier):
        if not identifier:
            raise gr.Error("削除する生成結果を選んでください。")
        try:
            with HISTORY_LOCK:
                delete_generation(OUTPUTS, identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except OSError as error:
            logging.exception("Generation deletion failed")
            raise gr.Error("生成結果を完全に削除できませんでした。出力先の権限を確認してください。") from error
        return None, None, "", None, gr.Group(visible=False), *refresh()

    with gr.Blocks(title="Persona Image Lab", fill_width=True) as app:
        gr.HTML(
            '<div id="studio-header"><h1>Persona Image Lab</h1>'
            '<p>キャラクターを選び、日本語で場面を入力して生成できます。</p></div>'
        )
        identifiers = gr.State([])
        selected_identifier = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5, min_width=340):
                persona = gr.Dropdown(
                    choices=persona_choices,
                    value="",
                    label="キャラクター",
                    info="Personaはこの端末の非公開ディレクトリから読み込みます。",
                )
                persona_status = gr.Markdown(persona_info(""), elem_id="persona-status")
                prompt = gr.Textbox(
                    label="プロンプト",
                    lines=6,
                    placeholder="例: 公園で立っている。白いワンピース、やわらかい夕方の光、全身。",
                    info="日本語の自然文で、場面・服装・表情・構図などをそのまま入力できます。",
                )
                with gr.Accordion("参照画像を追加（任意）", open=False):
                    refs = gr.File(
                        label="参照画像",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    gr.Markdown("衣装や構図など、今回だけ追加したい参考画像がある場合に使います。")
                size_preset = gr.Dropdown(
                    choices=list(size_presets),
                    value="正方形 · 1024 × 1024",
                    label="画像サイズ",
                )
                with gr.Accordion("詳細設定", open=False):
                    randomize_seed = gr.Checkbox(label="毎回違うSeedを使う", value=True)
                    with gr.Row(visible=False) as custom_dimensions:
                        width = gr.Slider(512, 2752, value=1024, step=32, label="幅")
                        height = gr.Slider(512, 2752, value=1024, step=32, label="高さ")
                    steps = gr.Slider(1, 80, value=40, step=1, label="生成ステップ")
                    seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=4294967295)
                create = gr.Button("画像を生成", variant="primary")
            with gr.Column(scale=7, min_width=420):
                result = gr.Image(label="生成結果", type="filepath", image_mode="RGBA", format="png",
                                  interactive=False, height=620, elem_id="result-panel")
                with gr.Row():
                    reuse = gr.Button("この画像を参照に追加")
                    delete = gr.Button("削除", variant="stop")
                stats = gr.Markdown("", visible=False, elem_id="generation-stats")
                with gr.Group(visible=False, elem_id="delete-confirmation") as delete_confirmation:
                    gr.Markdown("**この生成結果を完全に削除します。** 元に戻せません。")
                    with gr.Row():
                        cancel_delete = gr.Button("キャンセル")
                        confirm_delete = gr.Button("削除する", variant="stop")
        gallery = gr.Gallery(label="最近の生成", columns=5, height=300, preview=False,
                             elem_id="recent-generations")
        files = gr.File(label="PNG and generation record", file_count="multiple", visible=False)
        api_generate = gr.Button(visible=False)
        history = gr.Dataset(
            components=[prompt, gr.Textbox(render=False), width, height, steps, seed,
                        gr.Number(render=False)],
            headers=["Prompt", "Reference images", "Width", "Height", "Steps", "Seed", "Time (s)"],
            samples=[], type="index", layout="table", samples_per_page=10,
            label=None, visible=False,
        )
        refresh_outputs = [history, gallery, identifiers]
        restore_outputs = [persona, prompt, refs, size_preset, width, height, steps, seed,
                           randomize_seed, result, files, stats, selected_identifier]
        app.load(refresh, outputs=refresh_outputs, queue=False, api_name=False)
        persona.change(persona_info, inputs=persona, outputs=persona_status, queue=False, api_name=False)
        size_preset.change(custom_dimensions_visibility, inputs=size_preset,
                           outputs=custom_dimensions, queue=False, api_name=False)
        generation = create.click(run_with_progress,
                                  inputs=[persona, prompt, refs, size_preset, width, height,
                                          steps, seed, randomize_seed],
                                  outputs=[result, files, stats, *refresh_outputs, selected_identifier],
                                  concurrency_limit=1, show_progress_on=[result], api_name=False)
        generation.then(hide_delete_confirmation, outputs=delete_confirmation,
                        queue=False, api_name=False)
        api_generate.click(run,
                           inputs=[persona, prompt, refs, size_preset, width, height,
                                   steps, seed, randomize_seed],
                           outputs=[result, files, stats, *refresh_outputs, selected_identifier],
                           concurrency_limit=1, api_name="generate")
        selection = gallery.select(select_image, inputs=identifiers, outputs=restore_outputs,
                                   queue=False, api_name=False)
        selection.then(hide_delete_confirmation, outputs=delete_confirmation,
                       queue=False, api_name=False)
        history.click(restore, inputs=[history, identifiers], outputs=restore_outputs,
                      queue=False, api_name=False)
        reuse.click(use_reference, inputs=result, outputs=refs, queue=False)
        delete.click(request_delete, inputs=selected_identifier, outputs=delete_confirmation,
                     queue=False, api_name=False)
        cancel_delete.click(hide_delete_confirmation, outputs=delete_confirmation,
                            queue=False, api_name=False)
        confirm_delete.click(delete_selected, inputs=selected_identifier,
                             outputs=[refs, result, stats, selected_identifier,
                                      delete_confirmation, *refresh_outputs],
                             queue=False, api_name=False)
    return app.queue(max_size=8)


def serve():
    import gradio as gr

    prepare_output_directory()
    pipeline()
    build_app().launch(server_name=os.environ.get("PERSONA_HOST", "127.0.0.1"),
                       server_port=int(os.environ.get("PERSONA_PORT", "7860")), share=False,
                       allowed_paths=[str(OUTPUTS)], footer_links=[], run_history=False,
                       max_file_size="25mb",
                       theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
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

import argparse
import importlib.metadata
import json
import logging
import os
import queue
import secrets
import tempfile
import threading
import time
from pathlib import Path

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
from prompt_enhancer import (PromptEnhancerCancelled, PromptEnhancerError,
                             enhance_prompt)
from settings import (APP_VERSION, MODEL_DIR, MODEL_ID, MODEL_REVISION, OUTPUTS,
                      PROMPT_ENHANCER_ENABLED, ROOT, prepare_output_directory)

DEMOS = json.loads((ROOT / "demos.json").read_text())
LOCK = threading.Lock()
HISTORY_LOCK = threading.Lock()
PIPE = None
_GENERATION_CANCEL_LOCK = threading.Lock()
_ACTIVE_GENERATION_CANCEL = None


class GenerationCancelled(RuntimeError):
    pass


def _new_generation_cancel_event():
    global _ACTIVE_GENERATION_CANCEL
    event = threading.Event()
    with _GENERATION_CANCEL_LOCK:
        if _ACTIVE_GENERATION_CANCEL is not None:
            _ACTIVE_GENERATION_CANCEL.set()
        _ACTIVE_GENERATION_CANCEL = event
    return event


def _current_generation_cancel_event():
    with _GENERATION_CANCEL_LOCK:
        return _ACTIVE_GENERATION_CANCEL


def _request_generation_cancel():
    event = _current_generation_cancel_event()
    if event is None:
        return False
    event.set()
    return True


def _clear_generation_cancel_event(event):
    global _ACTIVE_GENERATION_CANCEL
    with _GENERATION_CANCEL_LOCK:
        if _ACTIVE_GENERATION_CANCEL is event:
            _ACTIVE_GENERATION_CANCEL = None
APP_CSS = """
.gradio-container {
  max-width: none !important;
  width: 100% !important;
  padding: 1rem clamp(0.75rem, 2vw, 2rem) 1.5rem !important;
}
#studio-header { padding: 0.15rem 0 0.8rem; }
#studio-header h1 { margin: 0 0 0.25rem; letter-spacing: -0.03em; }
#studio-header p { margin: 0; opacity: 0.7; }
#studio-shell { align-items: stretch; gap: clamp(0.8rem, 1.5vw, 1.4rem); }
#control-panel, #output-panel { min-width: 0; }
#persona-status { min-height: 2.4rem; opacity: 0.82; }
#result-panel { min-height: 68vh; }
#result-panel img { object-fit: contain !important; max-height: 76vh !important; }
#generation-stats {
  min-height: 2.4rem;
  padding: 0.55rem 0.75rem;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-md);
  background: var(--background-fill-secondary);
  opacity: 0.9;
}
#history-toolbar { align-items: center; margin-top: 0.75rem; }
#batch-selection-status { min-height: 1.5rem; opacity: 0.78; }
#delete-confirmation, #batch-delete-confirmation {
  width: min(100%, 440px);
  margin-left: auto;
  padding: 0.8rem 0.9rem;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}
#delete-confirmation p, #batch-delete-confirmation p { margin: 0 0 0.5rem; }
#delete-confirmation button, #batch-delete-confirmation button { min-height: 2.4rem; }
#recent-generations { margin-top: 0.75rem; }
#recent-generations .thumbnail-item { position: relative; }
#recent-generations .caption-label {
  position: absolute !important;
  top: 0.4rem !important;
  right: 0.4rem !important;
  bottom: auto !important;
  left: auto !important;
  display: grid !important;
  place-items: center;
  width: 1.7rem;
  height: 1.7rem;
  min-width: 1.7rem;
  padding: 0 !important;
  border: 1px solid rgba(255, 255, 255, 0.75);
  border-radius: 999px;
  background: rgba(20, 24, 31, 0.68) !important;
  color: white !important;
  font-size: 1rem;
  line-height: 1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.24);
  backdrop-filter: blur(4px);
}
#recent-generations .thumbnail-item:has(img[alt="☑"]) .caption-label {
  border-color: var(--color-accent);
  background: var(--color-accent) !important;
}
#result-panel button[aria-label="Share"],
#result-panel button[aria-label="共有"],
#recent-generations button[aria-label="Share"],
#recent-generations button[aria-label="共有"] { display: none !important; }
#recent-generations button[aria-label="Download All"],
#recent-generations button[aria-label="すべてダウンロード"] { display: none !important; }
@media (max-width: 640px) {
  .gradio-container { padding: 0.65rem !important; }
  #studio-header { padding-bottom: 0.5rem; }
  #result-panel { min-height: 48vh; }
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


def validate_manual_references(references):
    gradio_temp = Path(
        os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "gradio")
    ).resolve()
    allowed_roots = (Path(OUTPUTS).resolve(), gradio_temp)
    validated = []
    for reference in references or []:
        path = Path(reference)
        resolved = path.resolve()
        if (path.is_symlink() or not path.is_file()
                or not any(resolved.is_relative_to(root) for root in allowed_roots)):
            raise ValueError("参照画像はアップロード済み画像か保存済み生成結果から選んでください。")
        validated.append(str(resolved))
    return validated


def generate(prompt, references=None, width=1024, height=1024, steps=40, seed=42,
             context=None, user_prompt=None, progress_callback=None,
             enhancer=None, cancel_event=None):
    import torch

    prepare_output_directory()
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Generation was cancelled.")
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
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("Generation was cancelled.")
            step = step_index + 1
            if progress_callback is not None:
                progress_callback(step, steps)
            return callback_kwargs

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("Generation was cancelled.")
            image = pipe(
                prompt=prompt, image=images or None, width=width, height=height,
                num_inference_steps=steps, true_cfg_scale=1.0, use_kv_cache=True,
                generator=torch.Generator("cuda").manual_seed(seed),
                callback_on_step_end=(on_step_end
                                      if progress_callback is not None or cancel_event is not None
                                      else None),
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
    if enhancer:
        metadata["prompt_enhancer"] = enhancer
        if enhancer.get("rewritten_prompt"):
            metadata["rewritten_prompt"] = enhancer["rewritten_prompt"]
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
        "自動（Promptから判断）": "auto",
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
        context = entry.get("context")
        if isinstance(context, dict) and context.get("size_preset") == "自動（Promptから判断）":
            return "自動（Promptから判断）"
        dimensions = (entry["width"], entry["height"])
        for label, candidate in size_presets.items():
            if isinstance(candidate, tuple) and candidate == dimensions:
                return label
        return "カスタム"

    def dimensions_from_ratio(ratio):
        try:
            left, right = ratio.split(":", 1)
            value = float(left) / float(right)
        except (AttributeError, ValueError, ZeroDivisionError):
            return None
        if not 0.2 <= value <= 5:
            return None
        area = 1024 * 1024
        width = int((area * value) ** 0.5)
        height = int((area / value) ** 0.5)
        width = min(2752, max(512, round(width / 32) * 32))
        height = min(2752, max(512, round(height / 32) * 32))
        return width, height

    def enhancer_dimensions(enhanced, references):
        ratio = enhanced.get("wh_ratio", "")
        if ratio:
            return dimensions_from_ratio(ratio)
        follow = enhanced.get("ratio_follow", "")
        if follow.startswith("<image") and follow.endswith(">"):
            try:
                index = int(follow[6:-1]) - 1
                with Image.open(references[index]) as source:
                    ratio = f"{source.width}:{source.height}"
                return dimensions_from_ratio(ratio)
            except (IndexError, ValueError, OSError):
                return None
        return None

    def stats_text(stats):
        context = stats.get("context")
        persona_name = context.get("persona_name") if isinstance(context, dict) else None
        prefix = f"{persona_name} ・ " if persona_name else ""
        return (f"**完了** ・ {prefix}{stats.get('elapsed_seconds', 0):.1f}秒 ・ "
                f"{stats['width']} × {stats['height']} ・ Seed {stats['seed']}")

    def enhanced_prompt_update(entry):
        enhancer = entry.get("prompt_enhancer")
        if not isinstance(enhancer, dict) or not enhancer.get("enabled"):
            return gr.Textbox(value="", visible=False)
        value = enhancer.get("rewritten_prompt") or entry.get("rewritten_prompt")
        if not isinstance(value, str) or not value.strip():
            return gr.Textbox(value="", visible=False)
        return gr.Textbox(value=value, visible=True)

    def gallery_values(entries, selected_ids=None, selection_mode=False):
        selected = set(selected_ids or [])
        return [
            (entry['image_path'], ("☑" if entry['id'] in selected else "☐")
             if selection_mode else None)
            for entry in entries
        ]

    def refresh(selected_ids=None):
        entries = load_history(OUTPUTS)
        samples = [[display_prompt(r), ", ".join(ref.get('filename', 'Reference')
                                         for ref in r.get('references', [])),
                    r['width'], r['height'],
                    r['steps'], r['seed'], r.get('elapsed_seconds', 0)] for r in entries]
        return gr.Dataset(samples=samples), gallery_values(entries, selected_ids), [r['id'] for r in entries]

    def persona_info(identifier):
        if not identifier:
            return "キャラクター固定なし。必要なら参照画像を追加できます。"
        try:
            profile = get_persona(identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        description = profile.description or "ローカルPersona"
        return f"**{profile.name}** — {description}（参照画像 {len(profile.references)}枚）"

    def execute_generation(persona_id, prompt, refs, size_preset, width, height, steps, seed,
                           randomize_seed, use_enhancer=True,
                           status_callback=None, progress_callback=None,
                           enhanced_prompt_callback=None, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation was cancelled.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("プロンプトを入力してください。")
        profile = get_persona(persona_id)
        if size_preset not in size_presets:
            raise ValueError("利用できる画像サイズを選んでください。")
        dimensions = size_presets[size_preset]
        if isinstance(dimensions, tuple):
            width, height = dimensions
        elif dimensions == "auto":
            width, height = 1024, 1024
        if randomize_seed:
            seed = secrets.randbelow(2**32)
        persona_refs = [str(path) for path in profile.references] if profile else []
        references = persona_refs + validate_manual_references(refs)
        if len(references) > 10:
            raise ValueError("Personaと追加の参照画像は、合計10枚までです。")
        base_prompt = compose_prompt(profile, prompt)
        context = {"size_preset": size_preset}
        if profile:
            context.update({
                "persona_id": profile.identifier,
                "persona_name": profile.name,
                "persona_reference_count": len(persona_refs),
            })

        effective_prompt = base_prompt
        enhancer_metadata = {"enabled": False}
        if use_enhancer and PROMPT_ENHANCER_ENABLED:
            if status_callback:
                status_callback("短い指示を具体化しています…")
            try:
                enhanced = enhance_prompt(
                    base_prompt, references, seed=42, cancel_event=cancel_event
                )
                effective_prompt = enhanced["rewritten_prompt"]
                enhancer_metadata = {"enabled": True, **enhanced}
                if enhanced_prompt_callback:
                    enhanced_prompt_callback(effective_prompt)
                if dimensions == "auto":
                    suggested = enhancer_dimensions(enhanced, references)
                    if suggested is not None:
                        width, height = suggested
            except PromptEnhancerCancelled as error:
                raise GenerationCancelled("Generation was cancelled.") from error
            except PromptEnhancerError as error:
                logging.warning("Prompt enhancer failed after retry: %s", error)
                raise ValueError(
                    "Prompt自動補完に失敗しました。自動再試行でも復旧できなかったため、"
                    "画像生成を中止しました。もう一度生成するか、必要なら詳細設定から"
                    "『短い指示を自動で具体化』をOFFにしてください。"
                ) from error

        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Generation was cancelled.")
        if status_callback:
            status_callback("画像生成を準備中…")
        return generate(
            effective_prompt, references, width, height, steps, seed,
            context=context, user_prompt=prompt, progress_callback=progress_callback,
            enhancer=enhancer_metadata,
            cancel_event=cancel_event,
        )

    def run_with_progress(persona_id, prompt, refs, size_preset, width, height, steps, seed,
                          randomize_seed, use_enhancer):
        events = queue.Queue()
        cancel_event = _current_generation_cancel_event() or _new_generation_cancel_event()

        def worker():
            try:
                result = execute_generation(
                    persona_id, prompt, refs, size_preset, width, height, steps, seed,
                    randomize_seed, use_enhancer=use_enhancer,
                    status_callback=lambda message: events.put(("status", message)),
                    progress_callback=lambda step, total: events.put(("progress", step, total)),
                    enhanced_prompt_callback=lambda value: events.put(("enhanced_prompt", value)),
                    cancel_event=cancel_event,
                )
                events.put(("final", result))
            except GenerationCancelled:
                events.put(("cancelled", None))
            except BaseException as error:
                events.put(("error", error))
            finally:
                _clear_generation_cancel_event(cancel_event)

        threading.Thread(target=worker, daemon=True).start()
        generation_started = None
        while True:
            event = events.get()
            kind = event[0]
            if kind == "status":
                if event[1] == "画像生成を準備中…":
                    generation_started = time.perf_counter()
                yield (f"**{event[1]}**", gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                       gr.skip(), True, gr.skip(), gr.skip(), gr.skip())
                continue
            if kind == "progress":
                _, step, total = event
                started = generation_started or time.perf_counter()
                elapsed = time.perf_counter() - started
                if step < 2:
                    description = f"生成中 {step}/{total}"
                else:
                    remaining = (elapsed / step) * (total - step)
                    description = f"生成中 {step}/{total} ・ 残り約{remaining:.0f}秒"
                yield (f"**{description}**", gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                       gr.skip(), True, gr.skip(), gr.skip(), gr.skip())
                continue
            if kind == "enhanced_prompt":
                yield (gr.skip(), gr.Textbox(value=event[1], visible=True),
                       gr.skip(), gr.skip(), gr.skip(), gr.skip(), True, gr.skip(),
                       gr.skip(), gr.skip())
                continue
            if kind == "cancelled":
                yield ("**停止しました。生成結果は保存していません。**", gr.skip(),
                       gr.skip(), gr.skip(), gr.skip(), gr.skip(), False,
                       gr.Button(interactive=True), gr.Button(visible=False, interactive=True),
                       None)
                return
            if kind == "error":
                error = event[1]
                yield ("**生成に失敗しました**", gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                       gr.skip(), False, gr.Button(interactive=True),
                       gr.Button(visible=False, interactive=True), None)
                if isinstance(error, ValueError):
                    raise gr.Error(str(error)) from error
                if isinstance(error, TorchOutOfMemoryError):
                    raise gr.Error("GPUメモリが不足しました。画像サイズか参照画像の枚数を減らしてください。") from error
                if isinstance(error, (OSError, Image.DecompressionBombError)):
                    logging.exception("Image input or output failed", exc_info=error)
                    raise gr.Error("画像を読み込むか、生成結果を保存できませんでした。画像ファイルと空き容量を確認してください。") from error
                raise error
            image, metadata, stats = event[1]
            yield (stats_text(stats), enhanced_prompt_update(stats), *refresh(), stats["id"],
                   False, gr.Button(interactive=True),
                   gr.Button(visible=False, interactive=True),
                   {"image": image, "metadata": metadata})
            return

    def present_generation(completed):
        if not isinstance(completed, dict):
            return gr.skip(), gr.skip()
        image = completed.get("image")
        metadata = completed.get("metadata")
        if not image or not metadata:
            return gr.skip(), gr.skip()
        return image, [image, metadata]

    def run(persona_id, prompt, refs, size_preset, width, height, steps, seed, randomize_seed):
        try:
            image, metadata, stats = execute_generation(
                persona_id, prompt, refs, size_preset, width, height, steps, seed,
                randomize_seed, use_enhancer=True,
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

    def begin_generation():
        _new_generation_cancel_event()
        entries = load_history(OUTPUTS)
        return (True, "**生成を開始しています…**", gr.Textbox(value="", visible=False),
                None, False, [],
                gr.Markdown(value="", visible=False), gr.Button(visible=False),
                gr.Group(visible=False), gr.Button(interactive=True),
                gr.Button(value="選択"), gallery_values(entries),
                gr.Button(interactive=False), gr.Button(visible=True, interactive=True))

    def request_stop_generation():
        if not _request_generation_cancel():
            return "**停止対象の生成はありません。**", gr.Button(visible=False)
        return "**停止しています…**", gr.Button(visible=True, interactive=False)

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
                enhanced_prompt_update(entry), entry['id'])

    def toggle_batch_selection(index, identifiers, selected_ids):
        if not isinstance(index, int) or not 0 <= index < len(identifiers):
            raise gr.Error("生成履歴から画像を選んでください。")
        selected = set(selected_ids or [])
        identifier = identifiers[index]
        if identifier in selected:
            selected.remove(identifier)
        else:
            selected.add(identifier)
        return [identifier for identifier in identifiers if identifier in selected]

    def select_image(identifiers, generation_active, batch_mode, selected_ids,
                     event: gr.SelectData):
        if generation_active:
            gr.Info("生成中は履歴プレビューを固定しています。完了後に選択できます。")
            return (*([gr.skip()] * 14), selected_ids, gr.skip(), gr.skip(), gr.skip())
        if batch_mode:
            selected = toggle_batch_selection(event.index, identifiers, selected_ids)
            entries = load_history(OUTPUTS)
            status = f"**{len(selected)}件選択中**" if selected else "画像をクリックして選択します。"
            return (*([gr.skip()] * 14), selected,
                    gallery_values(entries, selected, selection_mode=True),
                    gr.Markdown(value=status, visible=True),
                    gr.Button(visible=bool(selected)))
        return (*restore(event.index, identifiers), selected_ids, gr.skip(), gr.skip(), gr.skip())

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

    def toggle_batch_mode(enabled):
        enabled = not enabled
        entries = load_history(OUTPUTS)
        status = "画像をクリックして選択します。" if enabled else ""
        return (enabled, [], gallery_values(entries, selection_mode=enabled),
                gr.Markdown(value=status, visible=enabled),
                gr.Button(visible=False), gr.Group(visible=False),
                gr.Button(interactive=not enabled),
                gr.Button(value="選択を終了" if enabled else "選択"))

    def request_batch_delete(selected_ids):
        count = len(selected_ids or [])
        if not count:
            raise gr.Error("削除する生成結果を選んでください。")
        return (gr.Markdown(value=f"**選択した{count}件を削除します。** 元に戻せません。"),
                gr.Group(visible=True),
                gr.Button(value="削除する" if count == 1 else f"{count}件を削除"))

    def delete_batch(selected_ids):
        identifiers_to_delete = list(dict.fromkeys(selected_ids or []))
        if not identifiers_to_delete:
            raise gr.Error("削除する生成結果を選んでください。")
        try:
            with HISTORY_LOCK:
                for identifier in identifiers_to_delete:
                    restore_generation(OUTPUTS, identifier)
                for identifier in identifiers_to_delete:
                    delete_generation(OUTPUTS, identifier)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except OSError as error:
            logging.exception("Batch generation deletion failed")
            raise gr.Error("生成結果を完全に削除できませんでした。出力先の権限を確認してください。") from error
        return (None, None, "", gr.Textbox(value="", visible=False), None, False, [],
                gr.Markdown(value="", visible=False),
                gr.Button(visible=False), gr.Group(visible=False),
                gr.Button(value="選択"), gr.Button(interactive=True), *refresh())

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
        return (None, None, "", gr.Textbox(value="", visible=False), None,
                gr.Group(visible=False), *refresh())

    with gr.Blocks(title="Persona Image Lab", fill_width=True) as app:
        gr.HTML(
            '<div id="studio-header"><h1>Persona Image Lab</h1>'
            '<p>キャラクターを選び、日本語で場面を入力して生成できます。</p></div>'
        )
        identifiers = gr.State([])
        selected_identifier = gr.State(None)
        generation_active = gr.State(False)
        completed_generation = gr.State(None)
        batch_selected = gr.State([])
        with gr.Row(elem_id="studio-shell"):
            with gr.Column(scale=4, min_width=340, elem_id="control-panel"):
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
                    value="自動（Promptから判断）",
                    label="画像サイズ",
                )
                with gr.Accordion("詳細設定", open=False):
                    use_enhancer = gr.Checkbox(
                        label="短い指示を自動で具体化",
                        value=PROMPT_ENHANCER_ENABLED,
                        interactive=PROMPT_ENHANCER_ENABLED,
                        info="参照画像なしはPE-T2I、Personaや参照画像ありはPE-I2Iを自動で使います。",
                    )
                    randomize_seed = gr.Checkbox(label="毎回違うSeedを使う", value=True)
                    with gr.Row(visible=False) as custom_dimensions:
                        width = gr.Slider(512, 2752, value=1024, step=32, label="幅")
                        height = gr.Slider(512, 2752, value=1024, step=32, label="高さ")
                    steps = gr.Slider(1, 80, value=40, step=1, label="生成ステップ")
                    seed = gr.Number(label="Seed", value=42, precision=0, minimum=0, maximum=4294967295)
                with gr.Row():
                    create = gr.Button("画像を生成", variant="primary")
                    stop_generation = gr.Button("停止", variant="stop", visible=False)
            with gr.Column(scale=8, min_width=480, elem_id="output-panel"):
                result = gr.Image(label="生成結果", type="filepath", image_mode="RGBA", format="png",
                                  interactive=False, height=720, elem_id="result-panel")
                with gr.Row():
                    reuse = gr.Button("この画像を参照に追加")
                    delete = gr.Button("削除", variant="stop")
                stats = gr.Markdown("待機中", visible=True, elem_id="generation-stats")
                enhanced_prompt = gr.Textbox(
                    label="具体化されたプロンプト",
                    lines=8,
                    interactive=False,
                    visible=False,
                    info="入力したプロンプトは変更せず、画像生成に使った具体化後の内容を表示します。",
                    elem_id="enhanced-prompt",
                )
                with gr.Group(visible=False, elem_id="delete-confirmation") as delete_confirmation:
                    gr.Markdown("**この画像を削除します。** 元に戻せません。")
                    with gr.Row():
                        cancel_delete = gr.Button("キャンセル")
                        confirm_delete = gr.Button("削除する", variant="stop")
        with gr.Row(elem_id="history-toolbar"):
            batch_mode = gr.State(False)
            batch_toggle = gr.Button("選択", scale=0, min_width=84)
            batch_status = gr.Markdown("", visible=False, elem_id="batch-selection-status")
            batch_delete = gr.Button("選択した画像を削除", variant="stop", visible=False)
        with gr.Group(visible=False, elem_id="batch-delete-confirmation") as batch_delete_confirmation:
            batch_delete_text = gr.Markdown("")
            with gr.Row():
                batch_cancel_delete = gr.Button("キャンセル")
                batch_confirm_delete = gr.Button("まとめて削除", variant="stop")
        gallery = gr.Gallery(label="最近の生成", columns=8, height=260,
                             allow_preview=False, preview=False,
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
                           randomize_seed, result, files, stats, enhanced_prompt,
                           selected_identifier]
        app.load(refresh, outputs=refresh_outputs, queue=False, api_name=False)
        persona.change(persona_info, inputs=persona, outputs=persona_status, queue=False, api_name=False)
        size_preset.change(custom_dimensions_visibility, inputs=size_preset,
                           outputs=custom_dimensions, queue=False, api_name=False)
        generation_start = create.click(begin_generation,
                                        outputs=[generation_active, stats, enhanced_prompt,
                                                 completed_generation, batch_mode,
                                                 batch_selected, batch_status, batch_delete,
                                                 batch_delete_confirmation, delete,
                                                 batch_toggle, gallery, create, stop_generation],
                                        queue=False, api_name=False)
        generation = generation_start.then(
            run_with_progress,
            inputs=[persona, prompt, refs, size_preset, width, height,
                    steps, seed, randomize_seed, use_enhancer],
            outputs=[stats, enhanced_prompt, *refresh_outputs, selected_identifier,
                     generation_active, create, stop_generation, completed_generation],
            concurrency_limit=1, show_progress="hidden", api_name=False,
        )
        generation.then(present_generation, inputs=completed_generation,
                        outputs=[result, files], queue=False, api_name=False)
        stop_generation.click(request_stop_generation,
                              outputs=[stats, stop_generation],
                              queue=False, api_name=False)
        generation.then(hide_delete_confirmation, outputs=delete_confirmation,
                        queue=False, api_name=False)
        api_generate.click(run,
                           inputs=[persona, prompt, refs, size_preset, width, height,
                                   steps, seed, randomize_seed],
                           outputs=[result, files, stats, *refresh_outputs, selected_identifier],
                           concurrency_limit=1, api_name="generate")
        selection = gallery.select(
            select_image,
            inputs=[identifiers, generation_active, batch_mode, batch_selected],
            outputs=[*restore_outputs, batch_selected, gallery, batch_status, batch_delete],
            queue=False, api_name=False,
        )
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
                             outputs=[refs, result, stats, enhanced_prompt, selected_identifier,
                                      delete_confirmation, *refresh_outputs],
                             queue=False, api_name=False)
        batch_toggle.click(toggle_batch_mode, inputs=batch_mode,
                           outputs=[batch_mode, batch_selected, gallery, batch_status,
                                    batch_delete, batch_delete_confirmation, delete,
                                    batch_toggle],
                           queue=False, api_name=False)
        batch_delete.click(request_batch_delete, inputs=batch_selected,
                           outputs=[batch_delete_text, batch_delete_confirmation,
                                    batch_confirm_delete],
                           queue=False, api_name=False)
        batch_cancel_delete.click(hide_delete_confirmation,
                                  outputs=batch_delete_confirmation,
                                  queue=False, api_name=False)
        batch_confirm_delete.click(
            delete_batch, inputs=batch_selected,
            outputs=[refs, result, stats, enhanced_prompt, selected_identifier,
                     batch_mode, batch_selected,
                     batch_status, batch_delete, batch_delete_confirmation, batch_toggle, delete,
                     *refresh_outputs],
            queue=False, api_name=False,
        )
    return app.queue(max_size=8)


def resolve_theme(gr):
    name = os.environ.get("PERSONA_THEME", "ocean").strip()
    builtins = {
        "soft": gr.themes.Soft,
        "ocean": gr.themes.Ocean,
        "monochrome": gr.themes.Monochrome,
        "glass": gr.themes.Glass,
    }
    factory = builtins.get(name.lower())
    if factory is not None:
        return factory()
    if "/" in name:
        try:
            return gr.Theme.from_hub(name)
        except Exception as error:
            logging.warning("Could not load Gradio theme %s: %s; using Ocean.", name, error)
    return gr.themes.Ocean()


def serve():
    import gradio as gr

    prepare_output_directory()
    pipeline()
    build_app().launch(server_name=os.environ.get("PERSONA_HOST", "127.0.0.1"),
                       server_port=int(os.environ.get("PERSONA_PORT", "7860")), share=False,
                       allowed_paths=[str(OUTPUTS)], footer_links=[], run_history=False,
                       max_file_size="25mb",
                       theme=resolve_theme(gr),
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

"""Official-style Qwen Image 2.1 prompt enhancement for interactive use."""

from dataclasses import dataclass
import base64
import io
import json
import threading
import time

from PIL import Image

from settings import (PE_I2I_DIR, PE_I2I_ID, PE_I2I_REVISION,
                      PE_I2I_SOCKET, PE_T2I_DIR, PE_T2I_ID, PE_T2I_REVISION,
                      PE_T2I_SOCKET, PROMPT_ENHANCER_BACKEND)


@dataclass(frozen=True)
class EnhancerProfile:
    task: str
    model_id: str
    revision: str
    directory: object
    takes_images: bool
    presence_penalty: float
    max_new_tokens: int
    socket: object


PROFILES = {
    "t2i": EnhancerProfile("t2i", PE_T2I_ID, PE_T2I_REVISION, PE_T2I_DIR,
                           False, 1.5, 768, PE_T2I_SOCKET),
    "i2i": EnhancerProfile("i2i", PE_I2I_ID, PE_I2I_REVISION, PE_I2I_DIR,
                           True, 0.0, 768, PE_I2I_SOCKET),
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_prompt": {"type": "string", "minLength": 1},
        "wh_ratio": {"type": "string"},
        "ratio_follow": {"type": "string"},
    },
    "required": ["rewritten_prompt", "wh_ratio", "ratio_follow"],
    "additionalProperties": False,
}

_CACHE = {}
_TOKENIZERS = {}
_LOCK = threading.Lock()


class PromptEnhancerError(RuntimeError):
    pass


class PromptEnhancerCancelled(PromptEnhancerError):
    pass


def choose_task(references):
    return "i2i" if references else "t2i"


def _split_thinking(text):
    if "</think>" in text:
        thinking, _, answer = text.partition("</think>")
        if "<think>" in thinking:
            thinking = thinking.partition("<think>")[2]
        return thinking.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _balanced_json_objects(text):
    spans = []
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:index + 1])
    return spans


def parse_answer(answer, task):
    answer = (answer or "").strip()
    for candidate in reversed(_balanced_json_objects(answer)):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        rewritten = payload.get("rewritten_prompt") or payload.get("rewrited_prompt")
        if isinstance(rewritten, str) and rewritten.strip():
            return {
                "rewritten_prompt": rewritten.strip(),
                "wh_ratio": str(payload.get("wh_ratio") or "").strip(),
                "ratio_follow": (str(payload.get("ratio_follow") or "").strip()
                                 if task == "i2i" else ""),
                "parse_ok": True,
            }
    return {"rewritten_prompt": "", "wh_ratio": "", "ratio_follow": "", "parse_ok": False}


def _load(profile):
    cached = _CACHE.get(profile.task)
    if cached is not None:
        return cached
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not (profile.directory / "system_prompt.txt").is_file():
        raise PromptEnhancerError(f"{profile.model_id} is not installed.")
    started = time.perf_counter()
    print(f"Loading {profile.model_id} in BF16 on CUDA", flush=True)
    processor = AutoProcessor.from_pretrained(str(profile.directory), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(profile.directory), dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda").eval()
    _CACHE[profile.task] = (processor, model)
    print(f"Prompt enhancer {profile.task} loaded in {time.perf_counter() - started:.1f}s", flush=True)
    return processor, model


def _presence_processor(penalty, prompt_length):
    if not penalty:
        return None
    from transformers import LogitsProcessor

    class PresencePenalty(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for batch in range(input_ids.shape[0]):
                generated = input_ids[batch, prompt_length:]
                if generated.numel():
                    scores[batch, generated.unique()] -= penalty
            return scores

    return PresencePenalty()


def _load_reference(path, max_pixels=1024 * 1024):
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    return image


def _image_data_url(path, max_pixels=512 * 512):
    image = _load_reference(path, max_pixels=max_pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _tokenizer(profile):
    cached = _TOKENIZERS.get(profile.task)
    if cached is not None:
        return cached
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(profile.directory), local_files_only=True)
    _TOKENIZERS[profile.task] = tokenizer
    return tokenizer


def _stream_vllm_response(client, endpoint, payload, task, cancel_event):
    payload = {**payload, "stream": True}
    parts = []
    finish_reason = None
    with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if cancel_event is not None and cancel_event.is_set():
                raise PromptEnhancerCancelled("Prompt enhancement was cancelled.")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            payload = json.loads(data)
            choice = payload.get("choices", [{}])[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if task == "t2i":
                fragment = choice.get("text", "")
            else:
                fragment = (choice.get("delta") or {}).get("content", "")
            if isinstance(fragment, str) and fragment:
                parts.append(fragment)
    if cancel_event is not None and cancel_event.is_set():
        raise PromptEnhancerCancelled("Prompt enhancement was cancelled.")
    return "".join(parts), finish_reason


def _vllm_request(profile, prompt, references, seed, *, max_tokens=None,
                  temperature=1.0, structured=True, cancel_event=None):
    import httpx

    socket = profile.socket
    if not socket.exists():
        raise PromptEnhancerError(f"Prompt enhancer {profile.task} service is not ready.")
    if cancel_event is not None and cancel_event.is_set():
        raise PromptEnhancerCancelled("Prompt enhancement was cancelled.")
    system_prompt = (profile.directory / "system_prompt.txt").read_text(encoding="utf-8").strip()
    transport = httpx.HTTPTransport(uds=str(socket))
    started = time.perf_counter()
    try:
        with httpx.Client(transport=transport, base_url="http://persona-pe", timeout=120.0) as client:
            if profile.task == "t2i":
                rendered = _tokenizer(profile).apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                payload = {
                    "model": "persona-pe-t2i",
                    "prompt": rendered,
                    "max_tokens": max_tokens or profile.max_new_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": profile.presence_penalty,
                    "seed": seed,
                }
                if structured:
                    payload["structured_outputs"] = {"json": OUTPUT_SCHEMA}
                if cancel_event is not None:
                    decoded, finish_reason = _stream_vllm_response(
                        client, "/v1/completions", payload, profile.task, cancel_event
                    )
                else:
                    response = client.post("/v1/completions", json=payload)
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    decoded = choice["text"]
                    finish_reason = choice.get("finish_reason")
            else:
                content = [
                    {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
                    for path in references
                ]
                content.append({"type": "text", "text": prompt})
                payload = {
                    "model": "persona-pe-i2i",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    "max_tokens": max_tokens or profile.max_new_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": profile.presence_penalty,
                    "seed": seed,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if structured:
                    payload["structured_outputs"] = {"json": OUTPUT_SCHEMA}
                if cancel_event is not None:
                    decoded, finish_reason = _stream_vllm_response(
                        client, "/v1/chat/completions", payload, profile.task, cancel_event
                    )
                else:
                    response = client.post("/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    decoded = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
    except PromptEnhancerCancelled:
        raise
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise PromptEnhancerError(f"Prompt enhancer {profile.task} service failed: {error}") from error
    return decoded, time.perf_counter() - started, finish_reason


def _transformers_request(profile, prompt, references, seed):
    import torch
    from transformers import LogitsProcessorList

    processor, model = _load(profile)
    system_prompt = (profile.directory / "system_prompt.txt").read_text(encoding="utf-8").strip()
    user_content = []
    for path in references:
        user_content.append({"type": "image", "image": _load_reference(path)})
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])
    prompt_length = inputs["input_ids"].shape[1]
    logits = LogitsProcessorList()
    presence = _presence_processor(profile.presence_penalty, prompt_length)
    if presence is not None:
        logits.append(presence)
    torch.manual_seed(seed)
    started = time.perf_counter()
    output = model.generate(
        **inputs, max_new_tokens=profile.max_new_tokens, do_sample=True,
        temperature=1.0, top_p=0.95, top_k=20,
        logits_processor=logits,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded = processor.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    return decoded, time.perf_counter() - started, None


def enhance_prompt(prompt, references=None, seed=42, cancel_event=None):
    references = references or []
    task = choose_task(references)
    profile = PROFILES[task]
    if profile.takes_images and not references:
        raise PromptEnhancerError("PE-I2I requires at least one reference image.")
    if not profile.takes_images and references:
        raise PromptEnhancerError("PE-T2I does not accept reference images.")

    backend = PROMPT_ENHANCER_BACKEND
    if backend not in {"auto", "vllm", "transformers"}:
        raise PromptEnhancerError(f"Unsupported prompt enhancer backend: {backend}")
    with _LOCK:
        if backend == "vllm" or (backend == "auto" and profile.socket.exists()):
            decoded, elapsed, finish_reason = _vllm_request(
                profile, prompt, references, seed, cancel_event=cancel_event
            )
            used_backend = "vllm"
        else:
            if cancel_event is not None and cancel_event.is_set():
                raise PromptEnhancerCancelled("Prompt enhancement was cancelled.")
            import torch
            with torch.inference_mode():
                decoded, elapsed, finish_reason = _transformers_request(
                    profile, prompt, references, seed
                )
            if cancel_event is not None and cancel_event.is_set():
                raise PromptEnhancerCancelled("Prompt enhancement was cancelled.")
            used_backend = "transformers"
    _, answer = _split_thinking(decoded)
    parsed = parse_answer(answer, task)
    retry_elapsed = 0.0
    retry_finish_reason = None
    if (not parsed["parse_ok"] or finish_reason == "length") and used_backend == "vllm":
        decoded, retry_elapsed, retry_finish_reason = _vllm_request(
            profile, prompt, references, seed,
            max_tokens=1536, temperature=0.2, structured=True,
            cancel_event=cancel_event,
        )
        _, answer = _split_thinking(decoded)
        parsed = parse_answer(answer, task)
    if not parsed["parse_ok"]:
        reason = retry_finish_reason or finish_reason or "unknown"
        raise PromptEnhancerError(
            f"Prompt enhancer output could not be parsed after retry (finish_reason={reason})."
        )
    return {
        **parsed,
        "task": task,
        "model": profile.model_id,
        "revision": profile.revision,
        "seed": seed,
        "backend": used_backend,
        "elapsed_seconds": round(elapsed + retry_elapsed, 2),
        "retried": bool(retry_elapsed),
        "finish_reason": retry_finish_reason or finish_reason,
    }

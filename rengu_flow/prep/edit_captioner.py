"""Edit-instruction captioning: the ``edit_caption`` prep stage.

An edit dataset pairs each target image with one or more control images (the trainer's
``control_path``; pairing rules in ``rengu_flow.data.control``). Its caption is not a
description but an *instruction*: the change that turns the controls into the target. This
stage shows a VLM the controls and then the target — all in one request, in that order — and
writes its instruction on line 1 of the target's caption (``CaptionStore``, so sidecar and
``captions.json`` layouts both work).

Inference is a llama.cpp ``llama-server`` over a GGUF VLM (``gguf_captioner``): one model load,
continuous batching across slots. Because every request carries several images, the server
context is sized from the dataset's largest row (:func:`gguf_captioner.server_budget`) instead
of the caption stage's fixed ``-c 32768 --parallel 16``, which leaves ~2048 tokens per slot and
would overflow on a control + target pair.

VLMs describing the difference between two images hallucinate changes (EditCaption,
arXiv 2604.08213): review the output in the caption editor before training on it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# The editable part of the prompt (the UI's default text). The image labels and the
# "image 1 / image 2" naming rule for multi-control rows are added around it by
# build_edit_content, so a custom prompt keeps the layout right.
DEFAULT_EDIT_PROMPT = (
    "Write the editing instruction that turns the source image into the result image. "
    "Use the imperative mood in one or two short sentences, for example "
    "\"Make the sky orange.\" or \"Remove the man on the left and add a red car.\" "
    "Describe only what changes; do not mention anything that stays the same. "
    "Output only the instruction, with no quotes, labels or preamble."
)

_MULTI_SOURCE_RULE = (
    "Images 1 to {n} are the source images and the last image is the result. "
    "When the instruction needs a source image, refer to it as \"image 1\", \"image 2\", etc."
)


def build_edit_content(n_controls: int, prompt: str | None = None) -> list[dict]:
    """The ordered user-turn parts for one row: labelled controls, the target, the prompt.

    Image parts are placeholders ``{"type": "image", "index": i}`` (0..n_controls, the target
    last) that the caller fills with the encoded image; text parts are final.
    """
    if n_controls < 1:
        raise ValueError("an edit row needs at least one control image")
    prompt = (prompt or "").strip() or DEFAULT_EDIT_PROMPT
    parts: list[dict] = []
    if n_controls == 1:
        parts += [{"type": "text", "text": "Source image:"}, {"type": "image", "index": 0}]
    else:
        for i in range(n_controls):
            parts += [
                {"type": "text", "text": f"Image {i + 1} (source):"},
                {"type": "image", "index": i},
            ]
    parts += [{"type": "text", "text": "Result image:"}, {"type": "image", "index": n_controls}]
    text = prompt if n_controls == 1 else f"{_MULTI_SOURCE_RULE.format(n=n_controls)}\n{prompt}"
    parts.append({"type": "text", "text": text})
    return parts


def render_edit_prompt(n_controls: int, prompt: str | None = None) -> str:
    """Human-readable form of :func:`build_edit_content` (images shown as ``<image>``)."""
    return "\n".join(
        "<image>" if p["type"] == "image" else p["text"]
        for p in build_edit_content(n_controls, prompt)
    )


_PREFIX_RE = re.compile(
    r"^\s*(?:here(?:'s| is)\s+(?:the|an|your)\s+)?(?:edit(?:ing)?\s+)?"
    r"(?:instruction|prompt)s?\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_QUOTES = "\"'`“”‘’«»"


def clean_instruction(text: str) -> str:
    """Normalize a VLM reply to a bare one-line instruction.

    Drops a ``<think>`` block, collapses lines, strips markdown emphasis, label prefixes
    (``Instruction:``, ``Edit instruction -``, ``Here is the instruction:``) and wrapping quotes.
    """
    from rengu_flow.prep.captioner import _collapse_to_one_line

    out = _collapse_to_one_line(text or "")
    out = out.replace("**", "").replace("__", "").strip()
    for _ in range(3):  # a label can sit inside quotes and vice versa
        prev = out
        out = _PREFIX_RE.sub("", out).strip()
        if len(out) >= 2 and out[0] in _QUOTES and out[-1] in _QUOTES:
            out = out[1:-1].strip()
        elif out and out[0] in _QUOTES and out.count(out[0]) == 1:
            out = out[1:].strip()  # unbalanced leading quote
        if out == prev:
            break
    return out


def edit_caption_folder(
    folder: str | Path,
    stage,
    *,
    fmt: str = "sidecar",
    ext: str = ".txt",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Write an edit instruction on line 1 of every paired target in ``folder``.

    ``stage`` is an ``EditCaptionStageConfig``. Targets without a valid control set are
    reported under ``unpaired`` (``"<image>: <reason>"``) and left untouched; targets whose
    line 1 already has text are skipped unless ``stage.overwrite``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rengu_flow.prep import gguf_captioner as gg
    from rengu_flow.prep.caption_store import CaptionStore

    cs = CaptionStore.open(folder, fmt=fmt, ext=ext, control_path=stage.control_path)
    spec = gg.get_gguf_model(stage.model)
    quant = gg.resolve_quant(spec, stage.gguf_quantization)

    paired = [k for k in cs.keys() if k in cs.controls]
    work = [k for k in paired if stage.overwrite or not (cs.get_lines(k)[:1] or [""])[0]]
    report: dict = {
        "captioned": 0,
        "skipped": len(paired) - len(work),
        "failed": [],
        "unpaired": [f"{k}: {reason}" for k, reason in sorted(cs.unpaired.items())],
        "model": spec.id,
        "quantization": quant,
        "stopped": False,
    }
    if cs.unpaired:
        logger.warning("edit_caption: %d target(s) have no valid controls", len(cs.unpaired))
    if not work:
        return report

    max_images = 1 + max(len(cs.controls[k]) for k in work)
    ctx_size, n_parallel = gg.server_budget(
        spec,
        n_images=max_images,
        max_pixels=stage.max_pixels,
        max_new_tokens=stage.max_new_tokens,
        n_parallel=stage.n_parallel or None,
    )
    report.update(ctx_size=ctx_size, n_parallel=n_parallel, max_images_per_row=max_images)
    logger.info(
        "edit_caption: %d pair(s), up to %d images/row at <=%d px -> -c %d --parallel %d",
        len(work), max_images, stage.max_pixels, ctx_size, n_parallel,
    )

    binary_dir = gg.ensure_binary()
    gguf, mmproj = gg.ensure_gguf(quant, model=spec.id)

    total = len(work)
    failed: list[str] = report["failed"]
    with gg.llama_server(binary_dir, gguf, mmproj, ctx_size=ctx_size, n_parallel=n_parallel) as port:

        def _one(key: str) -> tuple[str, Optional[str]]:
            images = [*cs.controls[key], cs.images[key]]
            try:
                encoded = [gg._encode_image(p, stage.max_pixels) for p in images]
                content = [
                    gg.image_part(encoded[p["index"]]) if p["type"] == "image" else p
                    for p in build_edit_content(len(images) - 1, stage.prompt)
                ]
                return key, gg.request_chat(
                    port, content, stage,
                    default_temperature=spec.default_temperature,
                    default_top_p=spec.default_top_p,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("edit_caption failed on %s: %s", key, exc)
                return key, None

        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futures = [pool.submit(_one, k) for k in work]
            for fut in as_completed(futures):
                if should_stop is not None and should_stop():
                    report["stopped"] = True
                    for f in futures:
                        f.cancel()
                    logger.info("edit_caption: stop signal after %d pairs", report["captioned"])
                    break
                key, text = fut.result()
                instruction = clean_instruction(text) if text is not None else ""
                if not instruction:
                    failed.append(key)
                else:
                    cs.set_line(key, 0, instruction)
                    report["captioned"] += 1
                    cs.save()  # incremental: a crash keeps everything done so far
                if on_progress is not None:
                    done = report["captioned"] + len(failed)
                    on_progress(done, total, f"edit-captioned {report['captioned']}/{total}")
    return report

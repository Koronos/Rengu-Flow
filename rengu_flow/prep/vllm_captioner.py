"""Standalone JoyCaption captioner — invoked via ``uv run --with vllm``.

vLLM pins its own torch (2.11 at time of writing) which is OLDER than the project's
torch 2.12, so it cannot share the project env without downgrading it. Hence this is
an **isolated** overlay subprocess (``uv run --with vllm``, no ``--project``): vLLM
brings its own torch/CUDA stack while the project .venv stays untouched. Contrast the
quality scorers (``aesthetic_scorer.py``), which DO reuse the project env because their
deps don't conflict.

Protocol: read a JSON manifest (path as argv[1]) describing the model + generation
params + a list of ``{key, image, prompt}`` items. Stream one result per line to
stdout as ``RESULT\\t{json}`` so the parent can save incrementally and show progress;
vLLM's own logs/progress bars go to stderr and are ignored by the parent.

Do NOT import this module from the package — it only ever runs as a subprocess entry
point inside the vLLM overlay env.
"""

import json
import sys
from pathlib import Path

RESULT_PREFIX = "RESULT\t"


def _resize(img, max_side: int):
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), __import__("PIL.Image", fromlist=["Image"]).LANCZOS)
    return img


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    items = manifest["items"]
    if not items:
        return 0

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    model = manifest["model"]
    # Pre-quantized checkpoints (gptq/awq) are auto-detected from the checkpoint config,
    # so only an on-the-fly scheme like fp8 needs to be passed explicitly.
    quant = manifest.get("quantization") or None
    llm_quant = quant if quant in ("fp8",) else None

    llm = LLM(
        model=model,
        quantization=llm_quant,
        max_model_len=manifest.get("max_model_len", 4096),
        gpu_memory_utilization=manifest.get("gpu_memory_utilization", 0.9),
        limit_mm_per_prompt={"image": 1},
        enforce_eager=manifest.get("enforce_eager", True),
    )
    processor = AutoProcessor.from_pretrained(model)
    system_prompt = manifest.get("system_prompt", "You are a helpful image captioner.")

    sampling = SamplingParams(
        temperature=manifest.get("temperature", 0.6),
        top_p=manifest.get("top_p", 0.9),
        max_tokens=manifest.get("max_new_tokens", 512),
    )
    max_side = manifest.get("max_image_side", 1536)

    def build_text(prompt: str) -> str:
        # JoyCaption's chat template takes STRING content (it calls .replace); the LLaVA
        # <image> placeholder is inlined and the PIL image rides multi_modal_data.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>\n{prompt}"},
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process in chunks so results stream out (incremental save + live progress) and
    # peak host RAM stays bounded; vLLM still continuous-batches within each chunk.
    chunk = max(1, int(manifest.get("chunk_size", 128)))
    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        inputs = []
        keys = []
        for it in batch:
            try:
                img = _resize(Image.open(it["image"]).convert("RGB"), max_side)
            except Exception as exc:  # noqa: BLE001 — report load failure, keep going
                sys.stdout.write(RESULT_PREFIX + json.dumps({"key": it["key"], "error": str(exc)}) + "\n")
                sys.stdout.flush()
                continue
            inputs.append({"prompt": build_text(it["prompt"]), "multi_modal_data": {"image": img}})
            keys.append(it["key"])
        if not inputs:
            continue
        outputs = llm.generate(inputs, sampling)
        for key, out in zip(keys, outputs):
            caption = out.outputs[0].text.strip() if out.outputs else ""
            sys.stdout.write(RESULT_PREFIX + json.dumps({"key": key, "caption": caption}) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

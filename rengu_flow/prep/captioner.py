"""Natural-language VLM captioners for the dataset-prep module.

Heavy imports (torch, transformers, bitsandbytes) are all lazy — isolated inside
methods so importing this module never triggers GPU initialisation.

Public API
----------
CaptionerConfig     -- dataclass driving the whole pipeline
CaptionBackend      -- protocol / base class for VLM backends
JoyCaptionBackend   -- Llama-3.1-8B LLaVA (fancyfeast/llama-joycaption-beta-one)
ToriiGateBackend    -- Qwen3.5-based VLM  (Minthy/ToriiGate-0.5)
build_prompt        -- pure function: build per-image prompt string from config + tags
caption_folder      -- top-level driver that handles batching, OOM retry, incremental save
list_caption_models -- return list of registered model ids
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# BACKENDS registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, dict] = {
    "joycaption-beta-one": {
        "cls": "JoyCaptionBackend",
        "repo_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "default_prompt": "Write a long descriptive caption for this image in a formal tone.",
    },
    "toriigate-0.5": {
        "cls": "ToriiGateBackend",
        "repo_id": "Minthy/ToriiGate-0.5",
        "default_prompt": "Give a long and detailed description of the picture.",
    },
}


def list_caption_models() -> list[str]:
    """Return registered model ids."""
    return list(_BACKENDS.keys())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CaptionerConfig:
    model: str = "joycaption-beta-one"
    quantization: str = "bf16"           # "bf16" | "int8" | "nf4"
    prompt: Optional[str] = None         # None -> per-model default
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    batch_size: int = 4
    use_tags_as_grounding: bool = True   # only ToriiGate uses it
    overwrite: bool = False              # if False, skip images that already have line 2
    # The trainer's bucketing does the real resize later, so raw datasets can carry
    # 8K originals (decode RAM; dynamic-resolution VLMs explode image-token counts)
    # or thumbnails (garbage captions). Cap the long side before the processor and
    # optionally skip too-small images entirely (0 disables the filter).
    max_image_side: int = 1536
    min_image_side: int = 0


# ---------------------------------------------------------------------------
# Prompt builder (pure function — easily unit-tested without GPU)
# ---------------------------------------------------------------------------


def build_prompt(config: CaptionerConfig, tags: Optional[list[str]] = None) -> str:
    """Return the user-facing prompt string for one image.

    For JoyCaption the tags are ignored (model doesn't use grounding).
    For ToriiGate, if use_tags_as_grounding is True and tags is non-empty,
    the tags are appended as a <tags>…</tags> block.
    """
    base_prompt = config.prompt or _BACKENDS.get(config.model, {}).get(
        "default_prompt", "Describe this image."
    )

    if config.model == "toriigate-0.5" and config.use_tags_as_grounding and tags:
        tags_str = ", ".join(tags)
        return (
            f"{base_prompt} Also here are booru tags for better understanding of the "
            f"picture, you can use them as reference: <tags>{tags_str}</tags>"
        )

    return base_prompt


# ---------------------------------------------------------------------------
# Backend protocol / base
# ---------------------------------------------------------------------------


class CaptionBackend:
    """Base class / protocol for VLM caption backends."""

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def caption_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def unload(self) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Helpers shared by backends
# ---------------------------------------------------------------------------


def _make_bnb_config(quantization: str):
    """Build a BitsAndBytesConfig for int8 or nf4 quantisation (lazy import)."""
    import torch
    from transformers import BitsAndBytesConfig  # type: ignore

    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"Unknown quantisation: {quantization!r}")


def _collapse_to_one_line(text: str) -> str:
    """Strip surrounding whitespace and collapse internal newlines to spaces."""
    return " ".join(text.splitlines()).strip()


class _HFVisionBackend(CaptionBackend):
    """Shared transformers plumbing: batched left-padded generation + VRAM lifecycle.

    Subclasses set ``model_key`` and implement ``_load_model_and_processor`` and
    ``_build_chat_text``. Generation batches the whole image list in one forward:
    the tokenizer pads on the LEFT so every row's completion starts at the same
    index and ``output[:, input_len:]`` slices cleanly for the entire batch.
    """

    model_key: str = ""

    def __init__(self, config: CaptionerConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None

    def _load_model_and_processor(self):  # pragma: no cover - subclass hook
        raise NotImplementedError

    def _build_chat_text(self, prompt: str) -> str:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def load(self) -> None:
        repo_id = _BACKENDS[self.model_key]["repo_id"]
        logger.info("Loading %s backend from %s", self.model_key, repo_id)
        self._model, self._processor = self._load_model_and_processor()
        self._model.eval()
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        logger.info("%s backend ready", self.model_key)

    def _quant_kwargs(self) -> dict:
        if self.config.quantization in ("int8", "nf4"):
            return {"quantization_config": _make_bnb_config(self.config.quantization)}
        if self.config.quantization != "bf16":
            raise ValueError(f"Unknown quantisation: {self.config.quantization!r}")
        return {}

    def caption_batch(
        self, images: list[Image.Image], prompts: list[str]
    ) -> list[str]:
        import torch  # lazy

        assert self._processor is not None and self._model is not None, "Call load() first"

        texts = [self._build_chat_text(prompt) for prompt in prompts]
        inputs = self._processor(
            images=images, text=texts, return_tensors="pt", padding=True
        ).to("cuda:0")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        gen_kwargs: dict = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        captions = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [_collapse_to_one_line(c) for c in captions]

    def unload(self) -> None:
        from rengu_flow.utils.common import empty_cuda_cache

        self._model = None
        self._processor = None
        empty_cuda_cache()


# ---------------------------------------------------------------------------
# JoyCaption backend
# ---------------------------------------------------------------------------


class JoyCaptionBackend(_HFVisionBackend):
    """Llama-3.1-8B LLaVA backend (fancyfeast/llama-joycaption-beta-one-hf-llava).

    bf16 weighs ~17 GB — it fits a 24 GB card because the job queue guarantees the
    GPU is exclusive while a prep job runs; int8/nf4 are for smaller cards or more
    KV-cache headroom at larger batch sizes.
    """

    model_key = "joycaption-beta-one"

    def _load_model_and_processor(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore

        repo_id = _BACKENDS[self.model_key]["repo_id"]
        processor = AutoProcessor.from_pretrained(repo_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            **self._quant_kwargs(),
        )
        return model, processor

    def _build_chat_text(self, prompt: str) -> str:
        # Official JoyCaption convo shape (model card): system + plain-string user turn.
        messages = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# ToriiGate backend
# ---------------------------------------------------------------------------


class ToriiGateBackend(_HFVisionBackend):
    """Qwen3.5-based anime-specialist backend (Minthy/ToriiGate-0.5, ~5B)."""

    model_key = "toriigate-0.5"

    def _load_model_and_processor(self):
        import torch
        from transformers import AutoProcessor  # type: ignore

        try:
            # transformers >= 5 dropped AutoModelForVision2Seq in favor of this.
            from transformers import AutoModelForImageTextToText as AutoVLM  # type: ignore
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM  # type: ignore

        repo_id = _BACKENDS[self.model_key]["repo_id"]
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoVLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            **self._quant_kwargs(),
        )
        return model, processor

    def _build_chat_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _default_backend_factory(config: CaptionerConfig) -> CaptionBackend:
    if config.model == "joycaption-beta-one":
        return JoyCaptionBackend(config)
    if config.model == "toriigate-0.5":
        return ToriiGateBackend(config)
    raise ValueError(
        f"Unknown model {config.model!r}. Known models: {list_caption_models()}"
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def caption_folder(
    folder: str | Path,
    config: CaptionerConfig,
    *,
    fmt: str = "sidecar",
    ext: str = ".txt",
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    backend_factory: Optional[Callable[[CaptionerConfig], CaptionBackend]] = None,
) -> dict:
    """Caption all images in *folder* using *config*.

    Returns
    -------
    dict with keys:
        captioned  int   -- images that received a new caption
        skipped    int   -- images skipped (already have line 2, overwrite=False)
        failed     list  -- keys that raised an error
        stopped    bool  -- True if should_stop() triggered early exit
    """
    from rengu_flow.prep.caption_store import CaptionStore
    from rengu_flow.utils.common import empty_cuda_cache

    folder = Path(folder)
    cs = CaptionStore.open(folder, fmt=fmt, ext=ext)

    factory = backend_factory if backend_factory is not None else _default_backend_factory
    backend: Optional[CaptionBackend] = None

    all_keys = cs.keys()
    # Partition into to-caption / skipped
    to_caption: list[str] = []
    skipped = 0
    for key in all_keys:
        lines = cs.get_lines(key)
        if not config.overwrite and len(lines) >= 2 and lines[1]:
            skipped += 1
        else:
            to_caption.append(key)

    total = len(to_caption)
    captioned = 0
    failed: list[str] = []
    skipped_small: list[str] = []
    stopped = False

    try:
        backend = factory(config)
        backend.load()

        batch_size = config.batch_size
        i = 0
        while i < len(to_caption):
            # Check for early stop between batches
            if should_stop is not None and should_stop():
                stopped = True
                logger.info("caption_folder: stop signal received after %d images", captioned)
                break

            batch_keys = to_caption[i : i + batch_size]
            batch_images: list[Image.Image] = []
            batch_prompts: list[str] = []

            for key in batch_keys:
                try:
                    img_path = cs.images[key]
                    img = Image.open(img_path).convert("RGB")
                    if config.min_image_side and min(img.size) < config.min_image_side:
                        logger.info(
                            "Skipping %s: %dx%d below min_image_side=%d",
                            key, *img.size, config.min_image_side,
                        )
                        skipped_small.append(key)
                        batch_images.append(None)
                        batch_prompts.append("")
                        continue
                    if config.max_image_side and max(img.size) > config.max_image_side:
                        # In-place downscale: caption quality is unchanged (VLM vision
                        # towers see <=1536px anyway) but decode RAM and image-token
                        # counts (dynamic-resolution models) stay bounded.
                        img.thumbnail(
                            (config.max_image_side, config.max_image_side),
                            Image.LANCZOS,
                        )
                    tags = cs.get_tags(key) if config.use_tags_as_grounding else None
                    prompt = build_prompt(config, tags)
                    batch_images.append(img)
                    batch_prompts.append(prompt)
                except Exception as exc:
                    logger.warning("Failed to load image %s: %s", key, exc)
                    failed.append(key)
                    batch_images.append(None)  # placeholder — filtered below
                    batch_prompts.append("")

            # Filter out load failures
            valid_indices = [
                j for j, img in enumerate(batch_images) if img is not None
            ]
            valid_images = [batch_images[j] for j in valid_indices]
            valid_prompts = [batch_prompts[j] for j in valid_indices]
            valid_keys = [batch_keys[j] for j in valid_indices]

            if not valid_images:
                i += len(batch_keys)
                continue

            # Run inference — with one OOM retry at halved batch
            captions: list[str] = []
            try:
                captions = backend.caption_batch(valid_images, valid_prompts)
            except Exception as exc:
                if "OutOfMemoryError" in type(exc).__name__ or "CUDA out of memory" in str(exc):
                    logger.warning(
                        "OOM on batch size %d — halving and retrying once", len(valid_images)
                    )
                    empty_cuda_cache()
                    half = max(1, len(valid_images) // 2)
                    # Process in two halves
                    try:
                        captions = backend.caption_batch(
                            valid_images[:half], valid_prompts[:half]
                        )
                        captions += backend.caption_batch(
                            valid_images[half:], valid_prompts[half:]
                        )
                    except Exception as exc2:
                        logger.error(
                            "OOM on retry (halved batch) — raising: %s", exc2
                        )
                        raise
                    # The halved size fits — keep it for the rest of the run instead of
                    # re-triggering the same OOM on every subsequent batch.
                    batch_size = max(1, half)
                    logger.info("Continuing with batch size %d", batch_size)
                else:
                    # Non-OOM failure: mark all as failed, continue
                    logger.error("Batch inference failed: %s", exc)
                    failed.extend(valid_keys)
                    i += len(batch_keys)
                    continue

            # Write captions (collapse multi-line model output to one line as safety net)
            for key, caption in zip(valid_keys, captions):
                cs.set_line(key, 1, _collapse_to_one_line(caption))
                captioned += 1

            # Incremental save after every batch
            cs.save()

            done_so_far = captioned + len(failed)
            if on_progress is not None:
                on_progress(
                    done_so_far,
                    total,
                    f"captioned {captioned}/{total}",
                )

            i += len(batch_keys)

    finally:
        if backend is not None:
            try:
                backend.unload()
            except Exception as exc:
                logger.warning("Backend unload failed: %s", exc)

    return {
        "captioned": captioned,
        "skipped": skipped,
        "skipped_small": skipped_small,
        "failed": failed,
        "stopped": stopped,
    }

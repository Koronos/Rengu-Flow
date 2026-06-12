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
        # Model-card recommended sampling.
        "default_temperature": 0.6,
        "default_top_p": 0.9,
    },
    "toriigate-0.5": {
        "cls": "ToriiGateBackend",
        "repo_id": "Minthy/ToriiGate-0.5",
        "default_prompt": "Give a long and detailed description of the picture.",
        # Official batch script: temperature 0.5, top_p left at 1.0.
        "default_temperature": 0.5,
        "default_top_p": 1.0,
    },
}


def list_caption_models() -> list[str]:
    """Return registered model ids."""
    return list(_BACKENDS.keys())


# ---------------------------------------------------------------------------
# Composable prompts: one base + stackable modifiers + outfit policy
# ---------------------------------------------------------------------------

# Curated for diffusion-training captions. Phrasing leans on JoyCaption's trained
# instruction set (base request first, hard constraints appended after — exactly the
# pattern its "extra options" were trained with); ToriiGate is instruction-tuned and
# follows the same text.
#
# The two training-critical mechanisms encoded here:
# - A caption that names the medium ("anime", "a photo of") anchors the style to the
#   text instead of letting the model learn it -> `medium_neutral` modifier.
# - Trigger absorption: whatever the caption DESCRIBES becomes separable from the
#   character's trigger name at generation time; whatever it OMITS gets absorbed into
#   the trigger. Hence `character_name` (never describe inherent traits — they live in
#   the name) and the `outfit` policy: "omit" bakes the default outfit into the
#   trigger, "describe" makes it swappable, "mixed" alternates per image so the model
#   gets both signals (default outfit retrievable AND swappable).

_NO_META = (
    "Your response will be used to train a text-to-image model, so avoid useless "
    "meta phrases like 'This image shows', 'You are looking at', or 'In this "
    "picture'; start directly with the content."
)

PROMPT_BASES: dict[str, dict] = {
    "descriptive-long": {
        "label": "Descriptive — long (default)",
        "description": "Long full-scene caption for t2i training.",
        "prompt": (
            "Write a long, detailed caption for this image as one paragraph. "
            "Describe the subjects and their appearance, clothing and accessories, "
            "expressions, poses and actions, then the setting and background "
            "elements, lighting, color palette, and composition. Use precise, "
            "objective language and describe only what is actually visible."
        ),
    },
    "concise": {
        "label": "Concise (2-4 sentences)",
        "description": "Short caption for tight token budgets.",
        "prompt": (
            "Write a caption of two to four sentences covering only the most "
            "important elements of this image: the main subject and their "
            "appearance, the action or pose, the setting, and the overall "
            "lighting and mood."
        ),
    },
    "character-focus": {
        "label": "Character focus",
        "description": "Centered on the main character (character LoRAs).",
        "prompt": (
            "Write a detailed caption for this image centered on the main person "
            "or character: their appearance, expression, pose, and action, with "
            "the level of physical detail the constraints below allow. Finish "
            "with one or two sentences about the setting, lighting, and "
            "composition. Use precise, objective language."
        ),
    },
    "style-focus": {
        "label": "Style focus",
        "description": "Prioritizes artistic style over content (style LoRAs).",
        "prompt": (
            "Describe the artistic style of this image in detail: the medium and "
            "technique, line work, brushwork or rendering, shading and lighting "
            "treatment, color palette, level of detail, composition, and overall "
            "aesthetic. Then summarize the subject matter in one or two sentences."
        ),
    },
}

PROMPT_MODIFIERS: dict[str, dict] = {
    "demographics": {
        "label": "Age / ethnicity / skin tone",
        "description": (
            "Includes apparent age, ethnic/regional origin, and skin tone when "
            "perceivable."
        ),
        "text": (
            "When people or characters are depicted, include their apparent age "
            "(or age range), apparent ethnicity or regional origin, and skin tone "
            "whenever these are perceivable — best-effort estimates are fine for "
            "stylized characters."
        ),
    },
    "medium_neutral": {
        "label": "Medium-neutral (cross-style)",
        "description": (
            "Never mentions the medium (photo/anime/render): for training anime "
            "models on realistic data and vice versa. Conceptually incompatible "
            "with the style-focus base."
        ),
        "text": (
            "STRICT RULE: never mention or hint at the medium, style, or rendering "
            "of the image. Do not use words like photo, photograph, photorealistic, "
            "realistic, anime, manga, cartoon, drawing, illustration, painting, "
            "artwork, render, 3D, CGI, screenshot, stylized, or animated, and do "
            "not compare the image to any medium. Describe the scene exactly as if "
            "the question of how it was made did not exist."
        ),
    },
    # Register matching: the model learns to respond to the vocabulary its captions
    # were written in. Ornate VLM prose trains a model that only "wakes up" for
    # ornate prompts — plain captions make plain user prompts work.
    "plain_language": {
        "label": "Plain English (prompt-register match)",
        "description": (
            "Simple, direct vocabulary — the way real users (or non-native "
            "speakers) actually prompt: the model responds to plain prompts "
            "without LLM embellishment."
        ),
        "text": (
            "Write in simple, plain English: use common everyday words and short, "
            "direct sentences, like a non-native English speaker would write. Say "
            "'long brown hair', 'wearing a red dress', 'standing in a kitchen' — "
            "never literary or ornate vocabulary such as 'cascading tresses', "
            "'adorned with', 'verdant', or 'a symphony of color'."
        ),
    },
    "objective_only": {
        "label": "Objective only (no quality words)",
        "description": (
            "No aesthetic judgments (beautiful/stunning/masterpiece): describes, "
            "never evaluates — avoids depending on quality-word prompts at "
            "inference."
        ),
        "text": (
            "Describe, never evaluate: do not use subjective quality or beauty "
            "words such as beautiful, stunning, gorgeous, breathtaking, "
            "masterpiece, high quality, or aesthetically pleasing, and do not "
            "comment on how good the image looks."
        ),
    },
    "composition_camera": {
        "label": "Shot type + camera angle",
        "description": (
            "Includes shot type (close-up/medium/wide), camera angle and vantage: "
            "makes framing promptable at inference."
        ),
        "text": (
            "State the shot type (extreme close-up, close-up, medium close-up, "
            "medium shot, cowboy shot, medium wide shot, wide shot, or extreme "
            "wide shot), the camera angle (eye-level, low-angle, high-angle, "
            "overhead, dutch angle), and the point of view if notable."
        ),
    },
    "explicit_language": {
        "label": "Explicit language (NSFW datasets)",
        "description": (
            "Direct anatomical language for sexual content, no euphemisms."
        ),
        "text": (
            "If there is any nudity or sexual content, describe it with direct, "
            "explicit anatomical language; do not use euphemisms, do not soften "
            "or skip it, and do not moralize about it."
        ),
    },
}

OUTFIT_MODES = ("describe", "omit", "mixed")

_OUTFIT_DESCRIBE = (
    "Describe their clothing, outfit, and accessories explicitly and in detail."
)
_OUTFIT_OMIT = (
    "Do NOT describe their clothing, outfit, or accessories at all; treat the "
    "outfit as part of who they are and leave it completely unmentioned."
)

DEFAULT_PROMPT_BASE = "descriptive-long"
DEFAULT_PROMPT_MODIFIERS = ("demographics",)


def _character_canon_text(name: str, canon: str) -> str:
    # Variant-aware mode: characters appear off-model (aged-up, alternate hairstyle,
    # meme body forms). Absorb only the canonical look into the trigger; deviations
    # MUST be described or they get wrongly absorbed and pollute the trigger.
    return (
        f"There is a specific character in this image: you MUST refer to them as "
        f"'{name}'. {name}'s canonical look is: {canon}. STRICT RULE: never "
        f"describe a trait that matches this canonical look — the name '{name}' "
        f"already implies it. However, if {name}'s appearance in THIS image "
        "deviates from the canonical look — a different hairstyle or hair color, "
        "an older or younger appearance, a different body type, an alternate "
        "form — you MUST describe that deviation explicitly."
    )


def _character_trigger_text(name: str) -> str:
    # Mirrors JoyCaption's trained options ("you must refer to them as {name}" and
    # "do NOT include information about people that cannot be changed"): inherent
    # traits stay inside the trigger name, everything else stays describable. The
    # GPU smoke showed the polite phrasing leaks ("She has long brown hair...");
    # the STRICT RULE pattern is the one the model actually obeys.
    return (
        f"There is a specific character in this image: you MUST refer to them as "
        f"'{name}'. STRICT RULE: never describe {name}'s unchangeable physical "
        f"traits — say nothing about their hair (color, length, or style), eye "
        f"color, facial features, skin tone, body type, or age; the name "
        f"'{name}' already implies all of them. Wherever you would normally "
        f"describe {name}'s appearance, describe only their expression, pose, "
        "and action instead."
    )


def _stable_choice(image_key: str) -> bool:
    """Deterministic per-image coin flip (same key -> same side, ~50/50 overall)."""
    import zlib

    return zlib.crc32(image_key.encode("utf-8")) % 2 == 0


# Inherent-trait clauses to scrub from captions when a character trigger is set.
# The VLM is told not to describe these, but quantized models still leak the most
# salient one ("She has long brown hair...") — a deterministic post-pass guarantees
# the absorption property regardless of model obedience.
_TRAIT_CLAUSE_RE = None


def _trait_clause_re():
    global _TRAIT_CLAUSE_RE
    if _TRAIT_CLAUSE_RE is None:
        import re

        _TRAIT_CLAUSE_RE = re.compile(
            r"\b("
            r"hair|hairstyle|bangs|ponytails?|twintails?|braids?|"
            r"(?:blue|brown|green|hazel|amber|gr[ae]y|red|golden|aqua|violet|"
            r"purple|pink|crimson|heterochromi\w+) eyes|eye color|eyes? (?:are|is) \w+|"
            r"skin tone|complexion|skinned|"
            r"years old|age of|in (?:his|her|their) (?:early |late |mid-?)?"
            r"(?:teens|twenties|thirties|forties|fifties)|"
            r"slender|petite|curvy|stocky|muscular|lanky|plump|"
            r"facial features|face shape"
            r")\b",
            re.IGNORECASE,
        )
    return _TRAIT_CLAUSE_RE


def scrub_trait_clauses(caption: str) -> str:
    """Remove clauses that describe inherent physical traits.

    Splits sentences into clauses (on commas and ' and ') and drops the clauses that
    match the trait patterns, keeping the rest of the sentence. Minor grammatical
    roughness is acceptable in training captions; leaked traits are not — anything
    the caption describes stops being absorbed into the trigger name.
    """
    import re

    pattern = _trait_clause_re()
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    kept_sentences = []
    for sentence in sentences:
        if not pattern.search(sentence):
            kept_sentences.append(sentence)
            continue
        body = sentence.rstrip(".!?")
        terminal = sentence[len(body):] or "."
        clauses = re.split(r",\s+|\s+and\s+", body)
        kept = [c for c in clauses if c.strip() and not pattern.search(c)]
        if not kept:
            continue
        rebuilt = ", ".join(kept) + terminal
        rebuilt = rebuilt[0].upper() + rebuilt[1:]
        kept_sentences.append(rebuilt)
    return " ".join(kept_sentences)


def compose_prompt(
    base: str = DEFAULT_PROMPT_BASE,
    modifiers: tuple[str, ...] | list[str] = DEFAULT_PROMPT_MODIFIERS,
    *,
    character_name: str = "",
    character_canon: str = "",
    outfit: str = "describe",
    image_key: str | None = None,
) -> str:
    """Assemble a caption prompt: base request + stacked constraint sentences.

    ``outfit`` only adds text when a ``character_name`` is set ("describe" is the
    bases' natural behavior otherwise). "mixed" resolves per image via a stable
    hash of ``image_key`` so the dataset carries both outfit signals.
    """
    if base not in PROMPT_BASES:
        raise ValueError(f"Unknown prompt base {base!r}; known: {list(PROMPT_BASES)}")
    if outfit not in OUTFIT_MODES:
        raise ValueError(f"Unknown outfit mode {outfit!r}; known: {OUTFIT_MODES}")
    unknown = [m for m in modifiers if m not in PROMPT_MODIFIERS]
    if unknown:
        raise ValueError(f"Unknown prompt modifier(s) {unknown}; known: {list(PROMPT_MODIFIERS)}")

    parts = [PROMPT_BASES[base]["prompt"]]
    # Registry order keeps composition deterministic regardless of input order.
    parts.extend(
        PROMPT_MODIFIERS[m]["text"] for m in PROMPT_MODIFIERS if m in set(modifiers)
    )
    if character_name.strip():
        resolved = outfit
        if outfit == "mixed":
            resolved = "describe" if _stable_choice(image_key or "") else "omit"
        parts.append(_OUTFIT_DESCRIBE if resolved == "describe" else _OUTFIT_OMIT)
        # Last constraint wins with instruction-tuned VLMs: the trait rule goes
        # after everything else so nothing re-licenses describing appearance.
        if character_canon.strip():
            parts.append(_character_canon_text(character_name.strip(), character_canon.strip()))
        else:
            parts.append(_character_trigger_text(character_name.strip()))
    parts.append(_NO_META)
    return " ".join(parts)


def list_prompt_options() -> dict:
    """Bases + modifiers + outfit modes for the UI/API."""
    return {
        "bases": [
            {"id": base_id, **base} for base_id, base in PROMPT_BASES.items()
        ],
        "modifiers": [
            {"id": mod_id, **mod} for mod_id, mod in PROMPT_MODIFIERS.items()
        ],
        "outfit_modes": list(OUTFIT_MODES),
        "default_base": DEFAULT_PROMPT_BASE,
        "default_modifiers": list(DEFAULT_PROMPT_MODIFIERS),
        "no_meta": _NO_META,
        "character_trigger_template": _character_trigger_text("{name}"),
        "outfit_texts": {"describe": _OUTFIT_DESCRIBE, "omit": _OUTFIT_OMIT},
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CaptionerConfig:
    model: str = "joycaption-beta-one"
    quantization: str = "bf16"           # "bf16" | "int8" | "nf4"
    prompt: Optional[str] = None         # custom prompt; overrides the composed one
    prompt_base: str = DEFAULT_PROMPT_BASE
    prompt_modifiers: tuple[str, ...] = DEFAULT_PROMPT_MODIFIERS
    character_name: str = ""             # trigger name; inherent traits stay in it
    # Canonical look (e.g. "aqua twin-tail hair, blue eyes, slim teenage build").
    # When set, only canon-matching traits are absorbed and DEVIATIONS from it are
    # described (aged-up, alternate hairstyle, meme forms); the hard trait scrubber
    # is disabled in this mode since it cannot tell deviation from canon.
    character_canon: str = ""
    outfit: str = "describe"             # describe | omit | mixed (needs character_name)
    # 1-based caption line to write (2 = the standard NL caption line). Use 3+ to add
    # extra caption VARIANTS (rengu treats each line as one) — e.g. line 2 absorbed
    # trigger caption + line 3 full description from a second job.
    target_line: int = 2
    max_new_tokens: int = 512
    # None = the model's own recommended sampling (per-model defaults in _BACKENDS).
    temperature: float | None = None
    top_p: float | None = None
    batch_size: int = 4
    # ToriiGate's hybrid linear-attention layers are not padding-invariant: padded
    # batches produce slightly different (still coherent) captions for the shorter
    # sequences. True = generate one image at a time (exact, ~2.5x slower).
    exact_generation: bool = False
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


# ToriiGate is trained on FIXED prompt formats (model card: "deviating from them is
# highly discouraged" — free-form instructions make it repeat/derail). These are the
# two plain-text formats from its official scripts/prompts.py, verbatim.
_TORII_FORMATS = {
    "long": (
        "Make a caption for given image with natural text. Use 2 to 5 paragraphs. "
        "Make your description long and vivid, mentioning all the details.\n"
    ),
    "short": (
        "The caption for image should be quite short without long purple prose and "
        "slop. Cover main objects and details.\n"
    ),
}


def _build_toriigate_prompt(
    config: "CaptionerConfig",
    tags: Optional[list[str]],
    image_key: Optional[str] = None,
) -> str:
    """Native ToriiGate user query: trained format + official grounding blocks.

    Mirrors make_user_query in the model's scripts/prompts.py: '# Captioning
    format:' + format text, '# Booru tags for the image\\n[...]', '# Characters on
    picture: ... make sure to use them: [...]'. Our modifiers/outfit policy ride an
    extra-requirements section — minimal deviation, validated on GPU.
    """
    c_type = "short" if config.prompt_base == "concise" else "long"
    parts = [f"# Captioning format:\n{_TORII_FORMATS[c_type]}"]

    extra = [
        PROMPT_MODIFIERS[m]["text"]
        for m in PROMPT_MODIFIERS
        if m in set(config.prompt_modifiers)
    ]
    if config.character_name.strip():
        resolved = config.outfit
        if resolved == "mixed":
            resolved = "describe" if _stable_choice(image_key or "") else "omit"
        extra.append(_OUTFIT_DESCRIBE if resolved == "describe" else _OUTFIT_OMIT)
        if config.character_canon.strip():
            extra.append(
                _character_canon_text(config.character_name.strip(), config.character_canon.strip())
            )
    if extra:
        parts.append("# Extra requirements:\n" + "\n".join(extra) + "\n")
    if config.use_tags_as_grounding and tags:
        parts.append(f"# Booru tags for the image\n[{', '.join(tags)}]\n")
    if config.character_name.strip():
        parts.append(
            "# Characters on picture:\nHere are names/tags for characters from the "
            f"picture, make sure to use them: [{config.character_name.strip()}].\n"
        )
    return "\n".join(parts)


def build_prompt(
    config: CaptionerConfig,
    tags: Optional[list[str]] = None,
    image_key: Optional[str] = None,
) -> str:
    """Return the user-facing prompt string for one image.

    Resolution order: explicit custom ``prompt`` > model-native composition.
    JoyCaption takes the instruction-composed prompt (it is instruction-flexible);
    ToriiGate takes its trained format + official grounding blocks instead.
    ``image_key`` feeds the per-image resolution of ``outfit="mixed"``.
    """
    if not config.prompt and config.model == "toriigate-0.5":
        return _build_toriigate_prompt(config, tags, image_key)

    base_prompt = config.prompt or compose_prompt(
        config.prompt_base,
        config.prompt_modifiers,
        character_name=config.character_name,
        character_canon=config.character_canon,
        outfit=config.outfit,
        image_key=image_key,
    )

    if config.model == "toriigate-0.5" and config.use_tags_as_grounding and tags:
        # Custom-prompt path: still ground with the official block format.
        return f"{base_prompt}\n\n# Booru tags for the image\n[{', '.join(tags)}]\n"

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
    # Hybrid linear-attention/conv models (ToriiGate's Qwen3.5 base) are not
    # padding-invariant: greedy A/B showed padded batches paraphrase the shorter
    # sequences (coherent, just not identical). When True, config.exact_generation
    # switches to per-image generation (no padding, ~2.5x slower).
    padding_sensitive: bool = False
    # Resize cap in pixels applied right before the processor (None = off). ToriiGate
    # was trained at ~1.0 Mpx (model card known issues).
    max_pixels: int | None = None

    def __init__(self, config: CaptionerConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if self.max_pixels and image.width * image.height > self.max_pixels:
            scale = (self.max_pixels / (image.width * image.height)) ** 0.5
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
        return image

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
        assert self._processor is not None and self._model is not None, "Call load() first"
        images = [self._prepare_image(img) for img in images]
        if self.padding_sensitive and self.config.exact_generation:
            results: list[str] = []
            for image, prompt in zip(images, prompts):
                results.extend(self._generate([image], [prompt]))
            return results
        return self._generate(images, prompts)

    def _generate(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        import torch  # lazy

        texts = [self._build_chat_text(prompt) for prompt in prompts]
        inputs = self._processor(
            images=images, text=texts, return_tensors="pt", padding=True
        ).to("cuda:0")
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        gen_kwargs: dict = {"max_new_tokens": self.config.max_new_tokens}
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            # Explicit pad id silences the per-batch "setting pad_token_id" notice.
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        backend_defaults = _BACKENDS.get(self.model_key, {})
        temperature = (
            self.config.temperature
            if self.config.temperature is not None
            else backend_defaults.get("default_temperature", 0.6)
        )
        top_p = (
            self.config.top_p
            if self.config.top_p is not None
            else backend_defaults.get("default_top_p", 0.9)
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        # clean_up_tokenization_spaces is a WordPiece-era step that corrupts BPE
        # output (strips spaces before punctuation) — transformers 5 warns about it.
        captions = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
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
        # use_fast=False: the fast (torchvision) image processor can't do the LANCZOS
        # resample this model's preprocessor_config asks for and silently substitutes
        # BICUBIC — the PIL backend reproduces the training-time preprocessing exactly.
        processor = AutoProcessor.from_pretrained(repo_id, use_fast=False)
        model = LlavaForConditionalGeneration.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
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
    padding_sensitive = True  # see _HFVisionBackend: exact_generation opts out of padding
    max_pixels = 1_000_000  # trained at ~1.0 Mpx (official scripts resize to this)

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
            dtype=torch.bfloat16,
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
    # 1-based target line -> 0-based index; line 1 is the tag line, so captions
    # start at index 1. Higher targets add caption variants (one per line).
    target_idx = max(1, config.target_line - 1)
    # Partition into to-caption / skipped
    to_caption: list[str] = []
    skipped = 0
    for key in all_keys:
        lines = cs.get_lines(key)
        if not config.overwrite and len(lines) > target_idx and lines[target_idx]:
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
                    prompt = build_prompt(config, tags, image_key=key)
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
                caption = _collapse_to_one_line(caption)
                if (
                    config.character_name.strip()
                    and not config.character_canon.strip()
                    and not config.prompt
                ):
                    # Guarantee trigger absorption even when the (quantized) VLM
                    # leaks an inherent trait despite the instruction. Skipped in
                    # canon mode: deviations from canon MUST survive the caption.
                    caption = scrub_trait_clauses(caption)
                cs.set_line(key, target_idx, caption)
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

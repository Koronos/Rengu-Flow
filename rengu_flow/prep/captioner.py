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
        "repo_id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
        "default_prompt": "Write a long descriptive caption for this image in a formal tone.",
        # Model-card recommended sampling.
        "default_temperature": 0.6,
        "default_top_p": 0.9,
    },
    "toriigate-0.5": {
        "repo_id": "Minthy/ToriiGate-0.5",
        "default_prompt": "Give a long and detailed description of the picture.",
        # Official batch script: temperature 0.5, top_p left at 1.0.
        "default_temperature": 0.5,
        "default_top_p": 1.0,
    },
}


# vLLM-ready JoyCaption checkpoints by quantization. 4-bit (gptq) fits a 16 GB card and is
# the fast path; fp8 is the data-free option; "none" is full bf16 (needs ~17 GB / a 24 GB
# card). No public AWQ exists, so "awq" requires an explicit vllm_model.
_VLLM_JOYCAPTION_REPOS: dict[str, str] = {
    "gptq": "NeoChen1024/llama-joycaption-beta-one-hf-llava-GPTQ-4bit-sym-autoround",
    "fp8": "NeoChen1024/llama-joycaption-beta-one-hf-llava-FP8-Dynamic",
    "none": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "awq": "",
}


def resolve_vllm_model(config: "CaptionerConfig") -> str:
    """Repo id for the vLLM JoyCaption run: explicit override, else by quantization."""
    if config.vllm_model.strip():
        return config.vllm_model.strip()
    repo = _VLLM_JOYCAPTION_REPOS.get(config.vllm_quantization, "")
    if not repo:
        raise ValueError(
            f"No default vLLM repo for quantization {config.vllm_quantization!r}; "
            "set caption.vllm_model to a checkpoint repo."
        )
    return repo


def list_caption_models() -> list[str]:
    """Return registered model ids."""
    return list(_BACKENDS.keys())


def captioner_config_from_stage(stage) -> "CaptionerConfig":
    """Map a CaptionStageConfig (TOML/UI shape) onto the engine config.

    Shared by the runner and the UI's prompt-preview endpoint so both always
    agree on what a job will actually send to the model.
    """
    return CaptionerConfig(
        model=stage.model,
        quantization=stage.quantization,
        prompt=stage.prompt or None,
        prompt_base=stage.prompt_base,
        prompt_modifiers=tuple(stage.prompt_modifiers),
        character_name=stage.character_name,
        character_canon=stage.character_canon,
        outfit=stage.outfit,
        target_line=stage.target_line,
        max_new_tokens=stage.max_new_tokens,
        temperature=stage.temperature,
        top_p=stage.top_p,
        exact_generation=stage.exact_generation,
        batch_size=stage.batch_size,
        use_tags_as_grounding=stage.use_tags_as_grounding,
        overwrite=stage.overwrite,
        max_image_side=stage.max_image_side,
        min_image_side=stage.min_image_side,
        engine=stage.engine,
        vllm_quantization=stage.vllm_quantization,
        vllm_model=stage.vllm_model,
        gpu_memory_utilization=stage.gpu_memory_utilization,
        vllm_max_model_len=stage.vllm_max_model_len,
        gguf_quantization=stage.gguf_quantization,
    )


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
        # Per-model sampling used when temperature/top_p are left blank — surfaced
        # so the UI can show the real numbers instead of a vague "model default".
        "sampling_defaults": {
            model_id: {
                "temperature": entry.get("default_temperature"),
                "top_p": entry.get("default_top_p"),
            }
            for model_id, entry in _BACKENDS.items()
        },
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
    # Inference engine. "hf" = in-process transformers (any model). "vllm" = isolated
    # overlay running vLLM with continuous batching + paged attention — much faster over a
    # whole folder, JoyCaption only. vLLM pins an older torch so it can't share the project
    # env; it runs as a `uv run --with vllm` subprocess (see vllm_captioner.py).
    # "hf" (any model) | "vllm" (JoyCaption only) | "gguf" (ToriiGate only, via llama.cpp).
    engine: str = "hf"
    # vLLM-only knobs. A pre-quantized 4-bit checkpoint (gptq/awq) fits a 16 GB card and is
    # the fast path; fp8 is the data-free fallback (no checkpoint, ~8.5 GB). The repo is
    # resolved from vllm_quantization unless vllm_model overrides it.
    vllm_quantization: str = "gptq"  # gptq | fp8 | awq | none
    vllm_model: str = ""             # repo override ("" = resolve from vllm_quantization)
    # Fraction of currently-FREE VRAM vLLM may use (measured at launch, not of total), so a
    # busy GPU (UI, other procs) doesn't get squeezed into OOM. 0.9 = 90% of free, 10% spare.
    gpu_memory_utilization: float = 0.9
    vllm_max_model_len: int = 4096
    # GGUF (llama.cpp) weight quantization for ToriiGate. Q8_0 ≈ lossless; drop to Q6_K/Q5_K_M/
    # Q4_K_M for less VRAM and more speed at some quality cost. See GGUF_QUANTS.
    gguf_quantization: str = "Q8_0"


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
    """Strip a leading <think> reasoning block, then collapse internal newlines to spaces.

    Reasoning VLMs (e.g. ToriiGate-0.5) emit a ``<think>…</think>`` chain-of-thought before the
    caption; ``</think>`` is not a special token so it survives ``skip_special_tokens``. Keep only
    the text after the final ``</think>`` (the caption); a truncated/unclosed ``<think>`` is dropped.
    """
    import re

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
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


def _prepare_batch(cs, batch_keys: list[str], config: CaptionerConfig):
    """Load + resize images and build prompts for one batch (CPU-only, no GPU).

    Returns ``(valid_keys, valid_images, valid_prompts, failed, skipped_small)``.
    Runs on the prefetch worker thread; it only READS from the CaptionStore (image
    paths, tag lines), while the main thread owns all writes (set_line/save) — safe
    under the GIL since the images map is fixed for the run and the keys are disjoint.
    """
    valid_keys: list[str] = []
    valid_images: list[Image.Image] = []
    valid_prompts: list[str] = []
    failed: list[str] = []
    skipped_small: list[str] = []
    for key in batch_keys:
        try:
            img = Image.open(cs.images[key]).convert("RGB")
            if config.min_image_side and min(img.size) < config.min_image_side:
                logger.info(
                    "Skipping %s: %dx%d below min_image_side=%d",
                    key, *img.size, config.min_image_side,
                )
                skipped_small.append(key)
                continue
            if config.max_image_side and max(img.size) > config.max_image_side:
                # In-place downscale: caption quality is unchanged (VLM vision towers
                # see <=1536px anyway) but decode RAM and image-token counts stay bounded.
                img.thumbnail((config.max_image_side, config.max_image_side), Image.LANCZOS)
            tags = cs.get_tags(key) if config.use_tags_as_grounding else None
            prompt = build_prompt(config, tags, image_key=key)
            valid_keys.append(key)
            valid_images.append(img)
            valid_prompts.append(prompt)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", key, exc)
            failed.append(key)
    return valid_keys, valid_images, valid_prompts, failed, skipped_small


def _caption_via_vllm(
    cs,
    to_caption: list[str],
    target_idx: int,
    config: CaptionerConfig,
    *,
    skipped: int,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Caption ``to_caption`` through the vLLM overlay subprocess (JoyCaption only).

    Prompts are composed here (the main process owns the CaptionStore + tags) and handed
    to ``vllm_captioner.py`` via a manifest. vLLM pins an older torch, so the overlay runs
    isolated (``uv run --with vllm``, no ``--project``) — see the dependency-isolation rule.
    Results stream back line by line for incremental save + live progress.
    """
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    if config.model != "joycaption-beta-one":
        raise ValueError(f"engine='vllm' supports only joycaption-beta-one, not {config.model!r}")
    if not shutil.which("uv"):
        raise FileNotFoundError("uv is not on PATH; required to run the vLLM overlay.")

    # Build the manifest: skip too-small images here (PIL .size is header-only/cheap) so the
    # report's skipped_small matches the in-process path; the overlay does the long-side cap.
    skipped_small: list[str] = []
    items: list[dict] = []
    for key in to_caption:
        path = cs.images[key]
        if config.min_image_side:
            try:
                with Image.open(path) as probe:
                    if min(probe.size) < config.min_image_side:
                        skipped_small.append(key)
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", key, exc)
                continue
        tags = cs.get_tags(key) if config.use_tags_as_grounding else None
        items.append({"key": key, "image": str(path), "prompt": build_prompt(config, tags, image_key=key)})

    total = len(items)
    captioned = 0
    failed: list[str] = []
    stopped = False
    if not items:
        return {"captioned": 0, "skipped": skipped, "skipped_small": skipped_small,
                "failed": failed, "stopped": False}

    manifest = {
        "model": resolve_vllm_model(config),
        "quantization": config.vllm_quantization,
        "max_model_len": config.vllm_max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature if config.temperature is not None else 0.6,
        "top_p": config.top_p if config.top_p is not None else 0.9,
        "max_image_side": config.max_image_side,
        "items": items,
    }
    script = Path(__file__).with_name("vllm_captioner.py")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as mf:
        json.dump(manifest, mf)
        manifest_path = mf.name

    # Isolated overlay (no --project): vLLM brings its own torch. Sanitize the env so the
    # nested `uv run` doesn't inherit this process's VIRTUAL_ENV/PYTHONPATH/UV_* and try to
    # import the project's (incompatible) packages.
    cmd = ["uv", "run", "--with", "vllm", "python", str(script), manifest_path]
    env = {k: v for k, v in os.environ.items()
           if k not in ("VIRTUAL_ENV", "PYTHONPATH") and not k.startswith("UV_")}

    from rengu_flow.prep.vllm_captioner import RESULT_PREFIX

    try:
        with tempfile.TemporaryFile(mode="w+") as errf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True, env=env)
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    if should_stop is not None and should_stop():
                        stopped = True
                        proc.terminate()
                        logger.info("vLLM captioner: stop signal received after %d images", captioned)
                        break
                    if not line.startswith(RESULT_PREFIX):
                        continue  # vLLM logs / progress bars — ignore
                    try:
                        rec = json.loads(line[len(RESULT_PREFIX):])
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("key")
                    if not key:
                        continue
                    if rec.get("error"):
                        logger.warning("vLLM failed on %s: %s", key, rec["error"])
                        failed.append(key)
                        continue
                    caption = _collapse_to_one_line(rec.get("caption", ""))
                    if (
                        config.character_name.strip()
                        and not config.character_canon.strip()
                        and not config.prompt
                    ):
                        caption = scrub_trait_clauses(caption)
                    cs.set_line(key, target_idx, caption)
                    captioned += 1
                    cs.save()  # incremental: a crash mid-run keeps everything done so far
                    if on_progress is not None:
                        on_progress(captioned + len(failed), total, f"captioned {captioned}/{total}")
            finally:
                proc.stdout.close()  # type: ignore[union-attr]
                rc = proc.wait()
                if rc != 0 and not stopped and captioned == 0:
                    errf.seek(0)
                    tail = errf.read()[-2000:]
                    raise RuntimeError(f"vLLM captioner exited with code {rc} and no output:\n{tail}")
    finally:
        try:
            os.unlink(manifest_path)
        except OSError:
            pass

    return {"captioned": captioned, "skipped": skipped, "skipped_small": skipped_small,
            "failed": failed, "stopped": stopped}


def _caption_via_gguf(
    cs,
    to_caption: list[str],
    target_idx: int,
    config: CaptionerConfig,
    *,
    skipped: int,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Caption ``to_caption`` through a llama.cpp ``llama-server`` (ToriiGate GGUF).

    Loads the model once and continuous-batches concurrent requests over its OpenAI
    endpoint. The binary (Vulkan) and GGUF are fetched on first use. ToriiGate only —
    the GGUF repo and prompt format are specific to it.
    """
    import socket
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rengu_flow.prep import gguf_captioner as gg

    if config.model != "toriigate-0.5":
        raise ValueError(f"engine='gguf' supports only toriigate-0.5, not {config.model!r}")

    # Skip too-small images here (cheap header read) so skipped_small matches the other paths.
    skipped_small: list[str] = []
    work: list[str] = []
    for key in to_caption:
        if config.min_image_side:
            try:
                with Image.open(cs.images[key]) as probe:
                    if min(probe.size) < config.min_image_side:
                        skipped_small.append(key)
                        continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read %s: %s", key, exc)
                continue
        work.append(key)

    total = len(work)
    captioned = 0
    failed: list[str] = []
    stopped = False
    if not work:
        return {"captioned": 0, "skipped": skipped, "skipped_small": skipped_small,
                "failed": failed, "stopped": False}

    binary_dir = gg.ensure_binary()
    gguf, mmproj = gg.ensure_gguf(config.gguf_quantization)

    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))  # 0 = OS assigns a free ephemeral port (never hard-code 8080)
            return s.getsockname()[1]

    # Start on a free port; if it loses the race for that port (closed before the server binds),
    # retry on a fresh one. Avoids a hard-coded port that a dev server might already hold.
    proc = port = None
    for attempt in range(3):
        port = _free_port()
        proc = gg._start_server(binary_dir, gguf, mmproj, port)
        try:
            gg._wait_health(port, proc, timeout=180.0 if attempt == 0 else 30.0)
            break
        except Exception as exc:  # noqa: BLE001
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if attempt == 2:
                raise
            logger.warning("llama-server start failed on port %d (%s); retrying", port, exc)
    try:

        def _one(key: str) -> tuple[str, Optional[str]]:
            tags = cs.get_tags(key) if config.use_tags_as_grounding else None
            prompt = build_prompt(config, tags, image_key=key)
            try:
                b64 = gg._encode_image(cs.images[key])
                return key, gg._request_caption(port, b64, prompt, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gguf caption failed on %s: %s", key, exc)
                return key, None

        with ThreadPoolExecutor(max_workers=gg.N_PARALLEL) as pool:
            futures = [pool.submit(_one, k) for k in work]
            for fut in as_completed(futures):
                if should_stop is not None and should_stop():
                    stopped = True
                    for f in futures:
                        f.cancel()
                    logger.info("gguf captioner: stop signal after %d images", captioned)
                    break
                key, text = fut.result()
                if text is None:
                    failed.append(key)
                    continue
                caption = _collapse_to_one_line(text)
                if (
                    config.character_name.strip()
                    and not config.character_canon.strip()
                    and not config.prompt
                ):
                    caption = scrub_trait_clauses(caption)
                cs.set_line(key, target_idx, caption)
                captioned += 1
                cs.save()  # incremental: a crash keeps everything done so far
                if on_progress is not None:
                    on_progress(captioned + len(failed), total, f"captioned {captioned}/{total}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    return {"captioned": captioned, "skipped": skipped, "skipped_small": skipped_small,
            "failed": failed, "stopped": stopped}


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

    # vLLM / GGUF run the whole folder through one persistent server (continuous batching),
    # so they bypass the in-process per-batch loop entirely.
    if config.engine == "vllm":
        return _caption_via_vllm(
            cs, to_caption, target_idx, config,
            skipped=skipped, on_progress=on_progress, should_stop=should_stop,
        )
    if config.engine == "gguf":
        return _caption_via_gguf(
            cs, to_caption, target_idx, config,
            skipped=skipped, on_progress=on_progress, should_stop=should_stop,
        )

    try:
        backend = factory(config)
        backend.load()

        batch_size = config.batch_size

        # Overlap CPU batch preparation (decode + resize + prompt build) with GPU
        # generation: generation is GPU-bound and preparation is CPU-bound, so a
        # one-batch-ahead prefetch thread hides the preparation cost almost entirely.
        from concurrent.futures import ThreadPoolExecutor

        def _submit(pool: ThreadPoolExecutor, start: int):
            keys = to_caption[start : start + batch_size]
            return pool.submit(_prepare_batch, cs, keys, config), start, len(keys)

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-prefetch") as pool:
            pending = _submit(pool, 0) if to_caption else None
            while pending is not None:
                future, start, n = pending
                # Check for early stop between batches
                if should_stop is not None and should_stop():
                    stopped = True
                    future.cancel()
                    logger.info("caption_folder: stop signal received after %d images", captioned)
                    break

                valid_keys, valid_images, valid_prompts, batch_failed, batch_small = future.result()
                failed.extend(batch_failed)
                skipped_small.extend(batch_small)

                # Queue the NEXT batch's CPU prep before running the GPU on this one.
                # If the OOM path below shrinks batch_size, this already-queued batch keeps
                # the old size (it just hits the same halving retry once) — a one-batch lag.
                next_start = start + n
                pending = _submit(pool, next_start) if next_start < len(to_caption) else None

                if not valid_images:
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

"""Qwen-Image 2.1 text conditioning.

Reproduces ``QwenImage21Pipeline._get_qwen_prompt_embeds`` of the reference pipeline
(diffusers ``main`` 6256aa7, ``pipelines/qwenimage21/pipeline_qwenimage21.py``).

Text-to-image (``encode_prompts``), prompts without condition images:

- the raw t2i chat template (not ``apply_chat_template``), empty prompts replaced by ``" "``;
- **left** padding for batches (the side the checkpoint was trained with);
- the conditioning is the input of the text decoder's final RMSNorm — captured with a forward
  hook that returns the norm's input (what ``hidden_states[-1]`` meant before transformers 5);
- the ``_drop_idx`` system-message tokens are dropped from each sample's valid tokens, and the
  batch is re-padded on the **right** (the transformer requires right padding).

The reference runs the full ``Qwen3VLForConditionalGeneration``; for text-only input that is
exactly its inner ``Qwen3VLTextModel`` (no position ids are passed on either path), which is all
rengu loads for text-to-image.

Image-conditioned (``encode_prompts_with_images``, the reference's ``ti2i`` branch): the ti2i
template with one ``<imageN><|vision_start|><|image_pad|><|vision_end|>`` placeholder per
condition image, RGBA images composited over white for the vision encoder, the processor's
``pixel_values`` / ``image_grid_thw`` / ``mm_token_type_ids`` fed to the full ``Qwen3VLModel``
(vision tower + deepstack + multimodal RoPE), the same pre-norm hook / ``_drop_idx`` / padding,
plus the per-token ``image_pad_mask`` (``True`` at the ``<|image_pad|>`` slots, 2x2 latents each).
The reference shares one image list across the batch; here row ``i`` carries its own list (the
processor receives the images in placeholder order, which is exactly the reference's flattening
when every row has the same list).
"""

from __future__ import annotations

import torch

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
# What the processor's chat template renders for the system turn alone; the reference derives
# ``_drop_idx`` by tokenizing exactly this (14 tokens with the Qwen3-VL tokenizer).
SYSTEM_MESSAGE = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
# The image-conditioned (ti2i) template: one placeholder per condition image before the prompt
# (``ti2i_template`` numbers the extra ones exactly like the reference).
IMAGE_PLACEHOLDER = "<image1><|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{IMAGE_PLACEHOLDER}{{}}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_PAD_TOKEN = "<|image_pad|>"


def drop_index(tokenizer) -> int:
    """Number of leading system-message tokens dropped from the encoder outputs."""
    return len(tokenizer(SYSTEM_MESSAGE).input_ids)


def text_model_of(text_encoder):
    """The ``Qwen3VLTextModel`` inside whatever wraps it (a ``LazyStreamedEncoder``, a
    ``Qwen3VLModel`` or a full ``Qwen3VLForConditionalGeneration``)."""
    module = text_encoder
    if hasattr(module, "load") and hasattr(module, "is_loaded"):  # LazyStreamedEncoder
        module = module.load()
    if hasattr(module, "model") and not hasattr(module, "layers"):  # ...ForConditionalGeneration
        module = module.model
    return getattr(module, "language_model", module)


@torch.no_grad()
def encode_prompts(
    text_encoder,
    tokenizer,
    prompts: list[str],
    device: torch.device | str,
    drop_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``prompts`` into right-padded ``(B, L, hidden)`` embeddings and a ``(B, L)`` bool
    mask of valid tokens (``L`` = longest prompt after dropping the system tokens)."""
    if drop_idx is None:
        drop_idx = drop_index(tokenizer)
    prompts = [" " if not p else p for p in prompts]
    texts = [PROMPT_TEMPLATE_T2I.format(p) for p in prompts]
    inputs = tokenizer(texts, padding=True, padding_side="left", return_tensors="pt").to(device)

    text_model = text_model_of(text_encoder)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False
        )
    finally:
        handle.remove()
    hidden_states = outputs.last_hidden_state

    valid = inputs.attention_mask.bool()
    split = [hidden_states[i][valid[i]][drop_idx:] for i in range(hidden_states.shape[0])]
    max_len = max(e.shape[0] for e in split)
    embeds = hidden_states.new_zeros((len(split), max_len, hidden_states.shape[-1]))
    mask = torch.zeros((len(split), max_len), dtype=torch.bool, device=hidden_states.device)
    for i, e in enumerate(split):
        embeds[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = True
    return embeds, mask


def ti2i_template(num_images: int) -> str:
    """The reference's ti2i template for ``num_images`` condition images (``{}`` = prompt)."""
    replace = IMAGE_PLACEHOLDER
    for i in range(2, num_images + 1):
        replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
    return PROMPT_TEMPLATE_TI2I.replace(IMAGE_PLACEHOLDER, replace)


def vision_image(img):
    """The copy of a condition image the vision encoder reads: RGBA composited over white (the
    checkpoint's training convention); anything else as-is. The VAE still gets all channels."""
    from PIL import Image as PILImage

    if not isinstance(img, PILImage.Image):
        img = PILImage.fromarray(img)
    if img.mode == "RGBA":
        white = PILImage.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        img = white
    return img


def vision_language_model(text_model, vision_model, config, shell=None):
    """A ``Qwen3VLModel`` whose ``language_model`` / ``visual`` are the given (already loaded)
    modules — the reference's ``text_encoder.model`` without copying any weight. ``shell`` (a
    previous return value) is reused when it already wraps these modules."""
    if shell is not None and shell.language_model is text_model and shell.visual is vision_model:
        return shell
    if shell is None:
        from accelerate import init_empty_weights
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        with init_empty_weights(include_buffers=False):
            shell = Qwen3VLModel._from_config(config)
    shell.language_model = text_model
    shell.visual = vision_model
    shell.eval()
    return shell


@torch.no_grad()
def encode_prompts_with_images(
    vlm,
    processor,
    prompts: list[str],
    images: list[list],
    device: torch.device | str,
    drop_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode ``prompts`` with their condition ``images`` (``images[i]`` = the PIL images of row
    ``i``, in order) through ``vlm`` (a ``Qwen3VLModel`` or ``Qwen3VLForConditionalGeneration``).

    Returns right-padded ``(B, L, hidden)`` embeddings, the ``(B, L)`` bool mask of valid tokens
    and the ``(B, L)`` bool ``image_pad_mask`` (``True`` at the vision slots)."""
    if drop_idx is None:
        drop_idx = drop_index(processor.tokenizer)
    if len(images) != len(prompts):
        raise ValueError(f"{len(prompts)} prompts but {len(images)} condition-image lists")
    prompts = [" " if not p else p for p in prompts]
    texts = [ti2i_template(len(row)).format(p) for p, row in zip(prompts, images)]
    condition_pil_list = [vision_image(img) for row in images for img in row]
    model_inputs = processor(
        text=texts, images=condition_pil_list, padding=True, padding_side="left", return_tensors="pt"
    ).to(device)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "pixel_values": model_inputs.pixel_values,
        "image_grid_thw": model_inputs.image_grid_thw,
    }
    if hasattr(model_inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

    base = getattr(vlm, "model", vlm)  # ...ForConditionalGeneration -> Qwen3VLModel (no LM head)
    text_model = getattr(base, "language_model", base)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        hidden_states = base(**forward_kwargs).last_hidden_state
    finally:
        handle.remove()

    img_token_id = processor.tokenizer.encode(IMAGE_PAD_TOKEN)[0]
    valid = model_inputs.attention_mask.bool()
    split = [hidden_states[i][valid[i]][drop_idx:] for i in range(hidden_states.shape[0])]
    pads = [(model_inputs.input_ids[i][valid[i]] == img_token_id)[drop_idx:] for i in range(len(split))]
    max_len = max(e.shape[0] for e in split)
    embeds = hidden_states.new_zeros((len(split), max_len, hidden_states.shape[-1]))
    mask = torch.zeros((len(split), max_len), dtype=torch.bool, device=hidden_states.device)
    image_pad_mask = torch.zeros_like(mask)
    for i, (e, pad) in enumerate(zip(split, pads)):
        embeds[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = True
        image_pad_mask[i, : pad.shape[0]] = pad
    return embeds, mask, image_pad_mask

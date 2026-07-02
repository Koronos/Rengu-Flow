# Krea 2 modeling code

`rengu_flow/model/krea2/dit.py` is vendored and adapted from the Hugging Face `diffusers`
project (`src/diffusers/models/transformers/transformer_krea2.py`, `main` branch — the class
is not yet in a released diffusers version). Text-template constants and the packing /
position-id helpers follow `src/diffusers/pipelines/krea2/pipeline_krea2.py`.

- Copyright 2026 Krea AI and The HuggingFace Team.
- License: Apache License 2.0 — <http://www.apache.org/licenses/LICENSE-2.0>

Adaptations for training in rengu-flow (attention-processor indirection removed, SDPA
attention, no PEFT mixin, no internal gradient checkpointing) are documented in the file
header. Module and parameter names are unchanged for checkpoint compatibility.

The Krea 2 model weights themselves are distributed by Krea under the Krea 2 Community
License (see the `krea/Krea-2-Raw` and `krea/Krea-2-Turbo` repositories); rengu-flow does
not redistribute them.

`assets/qwen3vl_4b/` bundles the Qwen3-VL-4B-Instruct tokenizer and model config
(Copyright Alibaba Cloud, Apache License 2.0 — from `Qwen/Qwen3-VL-4B-Instruct`) so the
text encoder loads from a bare weights file with no network access.
`assets/qwen_image_vae_config.json` is the Qwen-Image VAE config (Apache License 2.0,
from the `krea/Krea-2-Raw` release). The single-file key conversion in `loading.py`
follows ComfyUI's Krea 2 implementation (`comfy/ldm/krea2/model.py`, GPL-3.0 project —
only the key-name correspondence was derived from it, no code was copied).

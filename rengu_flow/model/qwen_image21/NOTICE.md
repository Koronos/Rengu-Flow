# Qwen-Image 2.1 modeling code

`rengu_flow/model/qwen_image21/dit.py` and `vae.py` are vendored and adapted from the Hugging
Face `diffusers` project, `main` branch at commit
[`6256aa7`](https://github.com/huggingface/diffusers/commit/6256aa7666cedd47443adc8f82da9a10e110b09c)
(PR [#14804](https://github.com/huggingface/diffusers/pull/14804)) — the classes are not yet in
a released diffusers version:

- `src/diffusers/models/transformers/transformer_qwenimage21.py` (`QwenImage21Transformer2DModel`)
- `src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py` (`AutoencoderKLQwenImage21`)

The latent packing and `img_mask` helpers follow
`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py`.

- Copyright 2026 Qwen-Image Team / The Qwen Team and The HuggingFace Team.
- License: Apache License 2.0 — <http://www.apache.org/licenses/LICENSE-2.0>

Adaptations for training in rengu-flow (attention-processor indirection and flex-attention path
removed, SDPA attention, no PEFT/LoRA-scale/cache mixins, `forward` split into
`prepare_inputs`/blocks/`finalize`, RoPE positions that skip padded text) are documented in the
file headers. Module and parameter names are unchanged for checkpoint compatibility.

The Qwen-Image 2.1 model weights are distributed by the Qwen team under the Qwen Research
License (see the `Qwen/Qwen-Image-2.1` repository); rengu-flow does not redistribute them.

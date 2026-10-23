# MOSS-Image modeling code

`rengu_flow/model/moss_image/dit.py` (the DiT: `RMSNorm`, `situ`, the RoPE table, `TextBlock`,
`TextProjection`, `Block`, `MOSSImageTransformer`), `preview_sampling.py` (the sigma schedule and
the CFG / block-skip sampling loop) and `text.py` (`encode_prompts`) are ported from the official
`pipeline_moss_image.py` that ships inside the Hugging Face repository
[`OpenMOSS-Team/MOSS-Image-Base`](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) at revision
`85975793d8b3d432f1b6e27f352abaa4c2a4c0d3` (identical copy in `transformer/`):

- `MOSSImageTransformer`, `_Block`, `_TextBlock`, `_TextProjection`, `_RoPE`, `_RMSNorm`, `_situ`,
  `_rotate`, `_attention` -> `dit.py`
- `MOSSImageScheduler.set_timesteps` (fixed shift 5.0, nearest-grid snapping) ->
  `preview_sampling.reference_sigmas`
- `MOSSImagePipeline.encode_prompt` and `__call__` (CFG in fp32, `uncond_skip_layers = (22,)`,
  VAE de-normalization) -> `text.py` and `preview_sampling.py`

- Copyright OpenMOSS-Team. License: Apache License 2.0 — <http://www.apache.org/licenses/LICENSE-2.0>
  (HF card front-matter and the repository's `LICENSE`).

Adaptations for training in rengu-flow (documented in the file headers): a batched forward over a
right-padded text batch with per-sample RoPE positions and key masking (the reference runs one
sample at a time at its true text length), `embed` / `Block` / `final` split so the pipeline layers
and the preview sampler reuse the blocks, and the activation dtype read from a norm weight so a
frozen fp8 / NF4 base works. Module and parameter names are unchanged for checkpoint compatibility.

`timesteps.py` implements the training sigma sampler from the formula of the MOSS-Image technical
report (Table 6, final stage, and Eq. 4); no code was taken from any other trainer.

Not copied from: ComfyUI's `comfy/ldm/moss_image` and `comfy/lora.py` (GPL-3.0) were used only to
confirm parameter-name and LoRA-key correspondence (`lora_unet_<path>`, `diffusion_model.<path>` and
bare `<path>` map to the official module paths; separate `q` / `k` / `v` keys load on the fused
Comfy-Org checkpoint too); KohakuMossTrainer was consulted for conventions only (default adapter scope,
community hyperparameters), no code.

`assets/tokenizer/` (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`),
`assets/text_encoder/config.json` and `assets/vae_config.json` are the `tokenizer/`,
`text_encoder/config.json` and `vae/config.json` files of that Hugging Face revision (Apache-2.0
metadata and tokenizer vocabulary, no weights), bundled so single-file checkpoints load without the
full download. The bundled Qwen3.5-9B text encoder keeps the Qwen license of its own repository;
rengu-flow does not redistribute any weights.

# Networks (developer guide)

This document describes the **networks** package: where adapter logic lives, how to add a new network type for SDXL, and how to support a new model.

## Layout

- **`renga_flow/networks/`**
  - **`factorization.py`**: Integer factorization for LoKr (Kronecker) decomposition. No external dependency; used by the vendored LoKr path.
  - **`lora_sdxl.py`**: LoRA for SDXL. Functions: `configure(...)`, `save(...)`, `load(pipeline, adapter_path)`. Fuse is done via diffusers `fuse_lora()` or PEFT `merge_and_unload()` in the pipeline.
  - **`lokr_sdxl.py`**: LoKr for SDXL. Same API as lora_sdxl, plus `fuse(pipeline)` and `infer_lokr_config_from_state(state)`. Uses **LyCORIS** (`lycoris-lora`) when installed, else a **vendored** implementation (same math as diffusion-pipe/Comfy). Only the vendored path supports `fuse`.
  - **`adapter_dit.py`**: LoRA / LoKr for Cosmos Predict2 DiT (`configure`, `save`, `load_weights`). LoKr reuses `lokr_sdxl._apply_lokr_vendored`; save adds `.alpha` and `diffusion_model.*` prefix (diffusion-pipe local diff).

The SDXL pipeline in `renga_flow/model/sdxl.py` delegates to these modules: it does not implement LoRA/LoKr itself. It calls `networks.lora_sdxl` or `networks.lokr_sdxl` based on `config['adapter']['type']`.

**Config normalization (dim / alpha):** Before the adapter is configured, `renga_flow/config/defaults.py` normalizes the adapter config: if the user sets `dim` but not `rank`, it sets `adapter_config["rank"] = adapter_config["dim"]` (Kohya-style alias). For LoRA and LoKr it **forces** `alpha = rank` and **rejects** an explicit `alpha` in TOML (Comfy-compatible saves, same rule as diffusion-pipe `train.py`). The network modules (`lora_sdxl`, `lokr_sdxl`) always read `rank` and `alpha` from `adapter_config` in their `configure()`; they do not read `dim` (that is only for config/TOML). LoRA passes `alpha` as `lora_alpha` to PEFT’s `LoraConfig`; LoKr uses `alpha/rank` as the per-module scale (`_lokr_scale`).

## Contract

Each network module for a model (e.g. `lora_sdxl`, `lokr_sdxl`) is expected to provide:

- **`configure(unet, text_encoder, text_encoder_2, adapter_config)`**  
  Apply the adapter to the three modules (freeze base, inject adapter parameters, set `original_name` on parameters for the Saver). No return value.

- **`save(save_dir, state_dict, adapter_config)`**  
  Write the adapter weights to `save_dir` in the format expected by inference UIs (Kohya for LoRA, LyCORIS/Comfy for LoKr). `state_dict` keys use prefixes `unet.`, `text_encoder.`, `text_encoder_2.`.

- **`load(pipeline, adapter_path)`**  
  Load adapter weights from the directory `adapter_path` (which contains a `.safetensors` file) into the pipeline. For SDXL, `pipeline` is the renga `SDXLPipeline` instance (so `pipeline.unet`, `pipeline.text_encoder`, `pipeline.text_encoder_2` are the diffusers modules).

- **Fuse (load and merge into base)**  
  The pipeline method **`load_and_fuse_adapter(path)`** loads an adapter from `path` and fuses it into the base weights so the model has no separate adapter layers (inference-style). LoRA uses the diffusers pipeline’s `fuse_lora()` when available, otherwise PEFT’s `merge_and_unload()` on unet and text encoders. LoKr uses **`lokr_sdxl.fuse(pipeline)`**, which only supports the **vendored** backend; if LyCORIS is installed, `fuse` raises `NotImplementedError` with a message. For LoKr, if the adapter was not already configured, the pipeline infers a minimal config from the state dict via **`lokr_sdxl.infer_lokr_config_from_state(state)`**.

The **Saver** (`renga_flow/utils/saver.py`) collects parameters with `requires_grad` and `original_name` from the pipeline and passes the merged state dict to `model.save_adapter(save_dir, state_dict)`, which in turn calls the appropriate network’s `save`.

## Adding a new network for SDXL

**`[TODO]`** Example types not in tree today: `loha_sdxl`, etc.

1. Add a new module under `renga_flow/networks/`, e.g. **`[TODO]` `loha_sdxl.py`**, with `configure`, `save`, and `load` as above.
2. In `renga_flow/model/sdxl.py`:
   - In `configure_adapter`, add a branch for the new type (e.g. `elif self.adapter_type == "loha": networks_module.loha_sdxl.configure(...)`).
   - In `save_adapter`, add a branch that calls the new module’s `save`.
   - In `load_adapter_weights`, detect the new type (e.g. by config or by key names in the state dict) and call the new module’s `load`.
3. In `renga_flow/config/defaults.py`, add defaults for the new adapter type in the `if "adapter" in config` block.
4. In `renga_flow/config/validation.py`, add the new type to the allowed `adapter["type"]` values.

## Adding an existing network to another model

**`[TODO]`** All steps below for non-SDXL models.

To support LoRA (or LoKr) for a different model (e.g. Flux):

1. Add a new module, e.g. **`[TODO]` `renga_flow/networks/lora_flux.py`**, that implements the same contract but for the Flux pipeline (its root modules and state dict prefixes may differ).
2. In the Flux pipeline class, implement `configure_adapter`, `save_adapter`, and `load_adapter_weights` by delegating to `networks.lora_flux` (and optionally **`[TODO]` `networks.lokr_flux`** when implemented).
3. Ensure the pipeline sets `original_name` on all trainable parameters (or that the network module does it in `configure`) so the Saver can collect them.

## LyCORIS vs vendored LoKr

- **When `lycoris-lora` is installed**: `lokr_sdxl` uses `create_lycoris(..., algo="lokr")` and `apply_to()` on each of unet, text_encoder, text_encoder_2. Stored wrappers are used for save/load.
- **When not installed**: `lokr_sdxl` uses the vendored `_inject_lokr_into_linear` and `_apply_lokr_vendored` (same Kronecker logic and parameter names as Comfy/LyCORIS).
- Save format is the same in both cases (LyCORIS/Comfy), so adapters work in ComfyUI/Forge regardless of which backend was used.

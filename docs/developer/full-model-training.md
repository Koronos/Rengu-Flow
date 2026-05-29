# Full-model training (developer guide)

This page describes how **full-model finetuning** is implemented so you can extend it or add it for new models.

## How the mode is chosen

- **Adapter mode**: `config.get("adapter")` is truthy. The orchestrator calls `model.configure_adapter(adapter_config)` and sets `is_adapter = True`. Saving uses `save_adapter`; only adapter weights are written.
- **Full-model mode**: `config.get("adapter")` is absent or falsy. No adapter is configured; all (or a subset of) parameters remain trainable. Saving uses **`save_full_model`** → **`model.save_model(save_dir, state_dict)`**.

The decision is in `rengu_flow/main.py`: `is_adapter = bool(config.get("adapter"))`. The **Saver** (`rengu_flow/utils/saver.py`) branches on `is_adapter` in `save_model()`: adapter path calls `save_adapter(name)`, full-model path calls `save_full_model(name)`.

## Save path for full model

1. **Saver.save_full_model(name)**  
   - Collects parameters from the pipeline that have `original_name` (set by the model so keys match diffusers/Comfy prefixes).  
   - Gathers partial state dicts from all pipeline stages into a single `state_dict`, then calls **`model.save_model(save_dir, state_dict)`**.

2. **model.save_model(save_dir, diffusers_sd)**  
   - Implemented per model (e.g. SDXL in `rengu_flow/model/sdxl.py`).  
   - Splits `diffusers_sd` by prefix (`unet.`, `text_encoder.`, `text_encoder_2.`), converts to the target format (e.g. Comfy single-file), and writes the checkpoint (e.g. `model.safetensors`) plus VAE and text encoders.

So: full-model save is **save_full_model** → **model.save_model()**. No adapter state dict is involved.

## Freeze text encoders (SDXL)

When in full-model mode, the config option **`model.freeze_text_encoders`** can be set to train only the UNet:

- In **main**, after the adapter block, if there is no adapter and `config["model"].get("freeze_text_encoders", False)` is true, the orchestrator calls **`model.freeze_text_encoders()`**.
- **SDXL** implements `freeze_text_encoders()` by setting `requires_grad_(False)` on all parameters of `text_encoder` and `text_encoder_2`. Other models can implement it as a no-op or with their own logic.

The method is optional on the model contract (default no-op in `rengu_flow/model/base.py`).

## Block swap

**Training block swap** is implemented for adapter training on SDXL and Cosmos ([training-techniques.md](training-techniques.md)). Full-model configs must not set `blocks_to_swap` (validation and `main.py` enforce adapter-only).

## Dataset cache and VAE unload (SDXL)

SDXL cache hooks are implemented (see [Dataset and cache — model hooks](dataset-and-cache.md#model-hooks-for-cache)). After `DatasetManager.cache()`, unload behaviour for full-model SDXL:

- For **full fine tuning** (no adapter), the VAE must be kept on **CPU** (not moved to `meta`) when unloading submodels after caching. Otherwise the full checkpoint cannot be written at save time, because `model.save_model()` needs the VAE state dict. See diffusion-pipe `utils/dataset.py` around the unload loop (~L1155–1158): `if self.model.name == 'sdxl' and model is self.vae` and full fine tuning, then `model.to('cpu')` instead of `model.to('meta')`.

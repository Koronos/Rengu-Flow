# Dependencies and upstream (diffusion-pipe)

renga-flow does **not** depend on diffusion-pipe at runtime. The repo at `https://github.com/nicojanssens/diffusion-pipe` is a **behavior reference** for Cosmos Predict2 / Anima training.

## Git submodules in diffusion-pipe vs renga-flow

| Upstream piece | Used by `cosmos_predict2.py` train path? | renga-flow |
|--------------|------------------------------------------|------------|
| `submodules/ComfyUI` | No direct import in train script | **Not** added (GPL, heavy). LoKr math from `renga_flow/networks/lokr_sdxl.py`. |
| `submodules/Cosmos` (NVIDIA) | Only `models/cosmos.py` (other product) | **Not** added |
| `models/cosmos_predict2_modeling.py` | Yes | **In-repo** → `model/cosmos_predict2/dit.py` |
| `models/llm_adapter.py` | Yes | **In-repo** → `llm_adapter.py` |
| `models/wan/vae2_1.py` | Yes | **In-repo** → `wan_vae.py` |
| `configs/qwen3_06b`, `t5_old` | Yes | **In-repo** → `assets/` + `package-data` |
| `utils/offloading.py` | Only with `blocks_to_swap` | Out of austere scope |
| PyPI: einops, transformers, accelerate, torchvision | Yes | Optional extra `[cosmos_predict2]` |

## Local diffusion-pipe diffs to mirror (already ported)

1. **LoKr on DiT** — `base.py` inject/apply LoKr; renga-flow: `adapter_dit` + `lokr_sdxl`.
2. **`save_adapter` LoKr** — `.alpha` per module + `diffusion_model.` prefix; renga-flow: `adapter_dit.save`.
3. **LoKr TOML** — `alpha` from `rank`; explicit `alpha` rejected in defaults (same as SDXL).

## Already in renga-flow (no re-port)

| diffusion-pipe | renga-flow |
|----------------|------------|
| `utils/dynamic_loader.py` | `renga_flow/optim/resolver.py` |
| Huber / pseudo_huber | `model/loss_utils.py` |
| run_dir / local_rank | `main.py` |

## Licenses

See `renga_flow/model/cosmos_predict2/NOTICE.md` (NVIDIA Apache-2.0, Wan VAE header in `wan_vae.py`).

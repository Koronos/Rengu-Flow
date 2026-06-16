# Dependencies and upstream (diffusion-pipe)

rengu-flow does **not** depend on diffusion-pipe at runtime. A local clone of [diffusion-pipe](https://github.com/nicojanssens/diffusion-pipe) (or your fork) is a **behavior reference** for Cosmos Predict2 / Anima training.

## Git submodules in diffusion-pipe vs rengu-flow

| Upstream piece | Used by `cosmos_predict2.py` train path? | rengu-flow |
|--------------|------------------------------------------|------------|
| `submodules/ComfyUI` | No direct import in train script | **Not** added (GPL, heavy). LoKr math from `rengu_flow/networks/lokr_sdxl.py`. |
| `submodules/Cosmos` (NVIDIA) | Only `models/cosmos.py` (other product) | **Not** added |
| `models/cosmos_predict2_modeling.py` | Yes | **In-repo** → `rengu_flow/model/cosmos_predict2/dit.py` |
| `models/llm_adapter.py` | Yes | **In-repo** → `rengu_flow/model/cosmos_predict2/llm_adapter.py` |
| `models/wan/vae2_1.py` | Yes | **In-repo** → `rengu_flow/model/cosmos_predict2/wan_vae.py` |
| `configs/qwen3_06b`, `t5_old` | Yes | **In-repo** → `assets/` + `package-data` |
| `utils/offloading.py` | Only with `blocks_to_swap` | Ideas ported to [`rengu_flow/training/block_swap.py`](../../rengu_flow/training/block_swap.py); no full upstream fork |
| PyPI: einops, transformers, accelerate, torchvision | Yes | Optional extra `[cosmos_predict2]` |

## Local diffusion-pipe diffs to mirror (already ported)

1. **LoKr on DiT** — `base.py` inject/apply LoKr; rengu-flow: `adapter_dit` + `lokr_sdxl`.
2. **`save_adapter` LoKr** — `.alpha` per module + `diffusion_model.` prefix; rengu-flow: `adapter_dit.save`.
3. **LoKr TOML** — `alpha` from `rank`; explicit `alpha` rejected in defaults (same as SDXL).

## Already in rengu-flow (no re-port)

| diffusion-pipe | rengu-flow |
|----------------|------------|
| `utils/dynamic_loader.py` | `rengu_flow/optim/resolver.py` |
| Huber / pseudo_huber | `model/loss_utils.py` |
| run_dir / local_rank | `rengu_flow/main.py` |

## Licenses

- Project: `LICENSE` (GPL-3.0-or-later) and `THIRD_PARTY_NOTICES.md`.
- Cosmos in-repo: `rengu_flow/model/cosmos_predict2/NOTICE.md` (NVIDIA Apache-2.0, Wan VAE header in `wan_vae.py`).
- Vendor optimizers: `rengu_flow/vendor/diffusion_pipe_optimizers/NOTICE.md`.
- Upstream diffusion-pipe: GPL-3.0 (see THIRD_PARTY_NOTICES).

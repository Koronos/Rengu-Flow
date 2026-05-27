# Third-party notices

Renga Flow combines original code with material adapted from other projects. **This file is not legal advice.** If you redistribute or modify the project, keep these notices and comply with each component’s license.

## diffusion-pipe

- **Repository:** https://github.com/tdrussell/diffusion-pipe  
- **License:** GNU General Public License v3.0 (GPL-3.0)  
- **Use in renga-flow:** Design and portions of training flow; some code paths were reimplemented or copied into `renga_flow/` (see [dependencies-and-upstream.md](docs/developer/dependencies-and-upstream.md)).  
- **Vendor optimizers:** `renga_flow/vendor/diffusion_pipe_optimizers/` — see [NOTICE.md](renga_flow/vendor/diffusion_pipe_optimizers/NOTICE.md) (copied commit documented there).

## NVIDIA Cosmos Predict2 modeling

- **Files:** e.g. `renga_flow/model/cosmos_predict2/dit.py`, `llm_adapter.py`  
- **License:** Apache License 2.0  
- **Notice:** [renga_flow/model/cosmos_predict2/NOTICE.md](renga_flow/model/cosmos_predict2/NOTICE.md)

## Alibaba Wan VAE

- **File:** `renga_flow/model/cosmos_predict2/wan_vae.py`  
- **Copyright:** Alibaba Wan Team (see file header)

## AI Toolkit (automagic optimizer)

- **File:** `renga_flow/vendor/diffusion_pipe_optimizers/automagic.py`  
- **License:** MIT — Copyright (c) 2024 Ostris, LLC

## Test fixtures (CC0)

- **Path:** `tests/fixtures/smoke_cc0/`  
- **License:** CC0 1.0 — see [LICENSE](tests/fixtures/smoke_cc0/LICENSE) and [ATTRIBUTION.md](tests/fixtures/smoke_cc0/ATTRIBUTION.md)

## PyPI dependencies

Runtime libraries (PyTorch, DeepSpeed, diffusers, PEFT, etc.) are listed in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Each package has its own license on PyPI.

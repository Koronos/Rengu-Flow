# diffusion-pipe optimizers (vendored)

Code copied from [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) `optimizers/` and `optimizers/projectors/`.

**Source commit at copy time:** `535bc585391d7f7d861d5f8952f1e144bc997270`

## Files and upstream credit

| File | Notes |
|------|--------|
| `generic_optim.py` | Based on [sn-sm](https://github.com/timmytonga/sn-sm); Kahan summation for bfloat16; Muon from [Muon](https://github.com/KellerJordan/Muon) |
| `automagic.py` | Copied from [AI Toolkit](https://github.com/ostris/ai-toolkit) (MIT); Kahan summation additions |
| `adamw_8bit.py` | AdamW 8-bit with Kahan summation |
| `gradient_release.py` | Wrapper for per-parameter optimizer steps |
| `optimizer_utils.py` | Shared utilities (Auto8bitTensor, stochastic rounding) |
| `projectors/` | SVD / uniform / top-k projectors for GenericOptim |

See per-file headers for full license text. Summary in [NOTICE.md](NOTICE.md).

## Usage in renga-flow

Register aliases in TOML: `type = "genericoptim"`, `type = "automagic"`, `type = "adamw8bitkahan"`.

Optional dependencies: install renga-flow with `pip install -e ".[optim]"` for bitsandbytes, pytorch-optimizer, etc.

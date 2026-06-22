# Native Windows (no WSL)

Rengu Flow runs on **native Windows, single-GPU**, using the `accelerate` engine — plain
PyTorch, **no DeepSpeed**. This is separate from the [WSL workflow](../developer/wsl-windows-workflow.md),
which is still the path for multi-GPU and the DeepSpeed-only memory features.

## TL;DR

```powershell
# from the repo root, in PowerShell
.\rengu.cmd init            # or: uv sync     (creates .venv, installs the stack)
.\rengu.cmd train --config examples\minimal_config_lora_sdxl.toml
```

`uv` must be on `PATH` ([install uv](https://docs.astral.sh/uv/)). Everything else — a
compatible Python, PyTorch `2.12.0+cu130`, torchvision, and `triton-windows` — is installed by
`uv sync`. You do **not** need DeepSpeed, and you do **not** need WSL.

## What's required

- **NVIDIA GPU + driver** new enough for CUDA 13.x (`nvidia-smi` shows the CUDA version) —
  [**download NVIDIA drivers**](https://www.nvidia.com/Download/index.aspx).
- **uv** on `PATH` — [**install uv**](https://docs.astral.sh/uv/getting-started/installation/).
- **CUDA Toolkit 13.x is optional.** The `accelerate` engine never calls DeepSpeed, so `nvcc`
  is not needed for training. It only helps `torch.compile`, which also needs a C++ compiler.
  If you want compile: [**CUDA Toolkit 13.x**](https://developer.nvidia.com/cuda-downloads) and
  [**Visual Studio Build Tools**](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  ("Desktop development with C++"). Without compile, none of this is required.

`uv sync` pulls `triton-windows` automatically on Windows (the Triton build that `torch.compile`
needs — on Linux it comes bundled with torch). Its version tracks torch: `torch 2.12` → Triton 3.7.

## Choosing the engine

The backend is selected by `[training].engine` in `rengu.local.toml` (or the `RENGU_ENGINE`
env var, or the `engine` key in a run config). When unset it defaults **per OS**:

| `engine` | Driver | Default on |
|----------|--------|------------|
| `accelerate` | plain PyTorch, single-GPU, no DeepSpeed | **Windows** |
| `deepspeed` | DeepSpeed pipeline engine, multi-GPU | **Linux/WSL** |
| `accelerate_deepspeed` | Accelerate + DeepSpeed ZeRO | *(not implemented yet)* |

So on Windows you normally set nothing — `accelerate` is automatic. To force it explicitly:

```toml
[training]
engine = "accelerate"
```

## What works on native Windows

Everything in the normal training path runs on the `accelerate` engine:

- Training (SDXL LoRA/LoKr/LyCORIS/full-finetune, Cosmos Predict2), latent + text-embedding caching
- **Batching:** `gradient_accumulation_steps`, per-resolution `micro_batch_size_per_gpu`, `image_micro_batch_size_per_gpu`
- **Dataset scheduling:** resolution schedules / staged multi-res, caption variants
- **Memory:** `activation_offload`, `activation_checkpointing` (`true`/interval **and** `auto`)
- `torch.compile` (via `triton-windows`), EMA, OOM-skip, eval, generalization probe, previews
- Checkpoint save/resume, model/adapter export, the web UI

## What is Linux/WSL-only

These use DeepSpeed's pipeline engine and **raise a clear error** on the `accelerate` engine:

- **Multi-GPU** and **`pipeline_stages > 1`** — Windows has no NCCL.
- **`optimizer.gradient_release`** — rewrites DeepSpeed's pipeline instruction map.
- **`blocks_to_swap`** (block swapping) — patches the DeepSpeed engine.

If you need any of these, use Linux or WSL2 with `engine = "deepspeed"`.

## Notes

- **`torch.compile` first step is slow** (1–4 min of kernel compilation, then disk-cached) — this
  is normal, not a hang.
- **Hugging Face cache symlinks:** Windows needs Developer Mode (or admin) for symlinks; without
  it the HF cache uses copies (more disk). Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the
  warning.
- **Line endings:** the `rengu.cmd` launcher and `uv run rengu …` both bypass the bash `rengu`
  shebang, so the CRLF issues from the WSL workflow do not apply.

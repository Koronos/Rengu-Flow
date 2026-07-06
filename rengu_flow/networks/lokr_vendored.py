"""LoKr (LyCORIS Kronecker) adapter for SDXL. Uses LyCORIS if installed, else vendored implementation."""

import re
import types
from pathlib import Path

import safetensors
import torch
from torch import nn
import torch.nn.functional as F

from rengu_flow.networks.factorization import factorization
from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.save_io import atomic_save_safetensors

try:
    from lycoris import create_lycoris, LycorisNetwork
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vendored LoKr
# ---------------------------------------------------------------------------

def _lokr_factors(module):
    w1 = module.lokr_w1 if module._lokr_use_w1_full else module.lokr_w1_a @ module.lokr_w1_b
    w2 = module.lokr_w2 if module._lokr_use_w2_full else module.lokr_w2_a @ module.lokr_w2_b
    return w1, w2


def _lokr_delta_forward(self, x):
    """LoKr delta WITHOUT materializing kron(w1, w2) — the vec-trick.

    ``y = x @ kron(w1, w2)^T`` factors into two small GEMMs over the ``(in1, in2)`` /
    ``(out1, out2)`` grids. The old path built the dense ``[out_dim, in_dim]`` delta on
    EVERY call (and again in the AC recompute): at full_matrix factor=6 on krea2 that
    was a base-weight-sized buffer plus a base-sized extra GEMM per linear — measured
    2x step time and the OOM margin. This form costs ~1/6th the FLOPs at those shapes
    and never allocates an ``[out_dim, in_dim]`` tensor. Runs in the activation dtype;
    fp32 masters are untouched.
    """
    out1, out2, in1, in2, out_dim, _in_dim = self._lokr_dims
    w1, w2 = _lokr_factors(self)
    w1 = w1.to(x.dtype)
    w2 = w2.to(x.dtype)
    shp = x.shape
    grid = x.reshape(-1, in1, in2)                     # [B, in1, in2]
    t = torch.matmul(grid, w2.transpose(0, 1))         # [B, in1, out2]
    y = torch.matmul(w1, t)                            # [B, out1, out2] (w1 broadcasts)
    return y.reshape(*shp[:-1], out_dim) * self._lokr_scale


def _lokr_forward_quantized(self, x):
    base = self._lokr_base_linear(x)
    return base + _lokr_delta_forward(self, x).to(base.dtype)


def _lokr_forward_plain(self, x):
    return F.linear(x, self.weight, self.bias) + _lokr_delta_forward(self, x).to(x.dtype)


def _inject_lokr_into_linear(module, rank, alpha, factor=-1, decompose_both=False, full_matrix=False, dtype=torch.float32):
    """Inject LoKr parameters into an nn.Linear and replace its forward. ComfyUI/LyCORIS convention."""
    weight = module.weight
    # Logical dims, not weight.shape: a bnb Linear4bit's weight is already packed
    # (out*in/2, 1) once quantized (eager since the block-swap fix), and factorizing
    # those dims builds a garbage delta. in/out_features are correct on both.
    if hasattr(module, "out_features") and hasattr(module, "in_features"):
        out_dim, in_dim = module.out_features, module.in_features
    else:
        out_dim, in_dim = weight.shape
    out1, out2 = factorization(out_dim, factor)
    in1, in2 = factorization(in_dim, factor)

    use_w1_full = True
    use_w2_full = full_matrix
    if not full_matrix:
        if decompose_both and rank < max(out1, in1) / 2:
            use_w1_full = False
        if rank >= max(out2, in2) / 2:
            use_w2_full = True
            if is_main_process():
                print(f"  LoKr: rank {rank} too large for W2 ({out2}x{in2}), using full matrix for W2")

    if use_w1_full:
        w1 = nn.Parameter(torch.empty(out1, in1, dtype=dtype))
        nn.init.kaiming_uniform_(w1, a=5**0.5)
        module.register_parameter("lokr_w1", w1)
    else:
        w1a = nn.Parameter(torch.empty(out1, rank, dtype=dtype))
        w1b = nn.Parameter(torch.empty(rank, in1, dtype=dtype))
        nn.init.kaiming_uniform_(w1a, a=5**0.5)
        nn.init.kaiming_uniform_(w1b, a=5**0.5)
        module.register_parameter("lokr_w1_a", w1a)
        module.register_parameter("lokr_w1_b", w1b)

    if use_w2_full:
        w2 = nn.Parameter(torch.zeros(out2, in2, dtype=dtype))
        module.register_parameter("lokr_w2", w2)
    else:
        w2a = nn.Parameter(torch.empty(out2, rank, dtype=dtype))
        w2b = nn.Parameter(torch.zeros(rank, in2, dtype=dtype))
        nn.init.kaiming_uniform_(w2a, a=5**0.5)
        module.register_parameter("lokr_w2_a", w2a)
        module.register_parameter("lokr_w2_b", w2b)

    module._lokr_use_w1_full = use_w1_full
    module._lokr_use_w2_full = use_w2_full
    module._lokr_scale = 1.0 if (use_w1_full and use_w2_full) else alpha / rank

    module.weight.requires_grad_(False)
    if module.bias is not None:
        module.bias.requires_grad_(False)

    # If the base linear was quantized (fp8 scaled_mm / bnb 4-bit), route the base matmul
    # through its quantized path and add the trainable LoKr delta as a separate output term:
    #   F.linear(x, W + diff) == base(x) + F.linear(x, diff)
    # This is the QLoRA-style composition and is required for 4-bit (packed weights cannot
    # be summed with a bf16 diff) and keeps fp8 matmul active under the adapter.
    base_linear = None
    try:
        from rengu_flow.training.quantize_dit import base_linear_of

        base_linear = base_linear_of(module)
    except Exception:
        base_linear = None

    module._lokr_dims = (out1, out2, in1, in2, out_dim, in_dim)
    module._lokr_base_linear = base_linear
    # Real bound methods (not per-instance closures): dynamo specializes on the shared
    # function code object with self-relative attribute sources, matching how PEFT and
    # Fp8TensorwiseLinear express their forwards.
    module.forward = types.MethodType(
        _lokr_forward_quantized if base_linear is not None else _lokr_forward_plain, module
    )


def _fuse_one_lokr_linear(module):
    """Fuse LoKr delta into base weight and restore standard nn.Linear.forward. In-place."""
    if not hasattr(module, "_lokr_scale"):
        return False
    use_w1_full = getattr(module, "_lokr_use_w1_full", True)
    use_w2_full = getattr(module, "_lokr_use_w2_full", False)
    scale = module._lokr_scale
    w1 = module.lokr_w1 if use_w1_full else (module.lokr_w1_a @ module.lokr_w1_b)
    w2 = module.lokr_w2 if use_w2_full else (module.lokr_w2_a @ module.lokr_w2_b)
    diff = torch.kron(w1, w2).to(module.weight.dtype).to(module.weight.device) * scale
    module.weight.data.add_(diff.reshape(module.weight.shape))
    module.forward = lambda x, m=module: F.linear(x, m.weight, m.bias)
    for name in list(module._parameters.keys()):
        if "lokr_" in name:
            del module._parameters[name]
    for attr in ("_lokr_use_w1_full", "_lokr_use_w2_full", "_lokr_scale"):
        if hasattr(module, attr):
            delattr(module, attr)
    return True


def _fuse_vendored_on_module(module):
    """Walk module and fuse every LoKr-injected nn.Linear. Returns number fused."""
    n = 0
    for _name, sub in module.named_modules():
        if isinstance(sub, nn.Linear) and hasattr(sub, "_lokr_scale"):
            _fuse_one_lokr_linear(sub)
            n += 1
    return n


def _apply_lokr_vendored(module, containers, adapter_config, state_dict_key_prefix=""):
    """Inject vendored LoKr into every nn.Linear under ``containers`` (in-place).

    Params are registered on each nn.Linear, so they live inside the tree ``to_layers()``
    flattens — DeepSpeed then places and trains them like the base weights.
    """
    rank = adapter_config["rank"]
    alpha = adapter_config["alpha"]
    factor = adapter_config.get("factor", -1)
    decompose_both = adapter_config.get("decompose_both", False)
    full_matrix = adapter_config.get("full_matrix", False)
    dtype = adapter_config["dtype"]

    count = 0
    for container in containers:
        if container is None:
            continue
        for sub in container.modules():
            if isinstance(sub, nn.Linear) and not hasattr(sub, "_lokr_scale"):
                _inject_lokr_into_linear(sub, rank, alpha, factor, decompose_both, full_matrix, dtype)
                count += 1

    for name, p in module.named_parameters():
        p.original_name = state_dict_key_prefix + name
        if p.requires_grad and dtype is not None:
            p.data = p.data.to(dtype)

    if is_main_process():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"LoKr (vendored): injected into {count} Linear modules")
        print(f"  trainable params: {trainable:,d} || all params: {total:,d} || trainable%: {100 * trainable / total:.4f}")
        if full_matrix and trainable > 50_000_000:
            # full_matrix stores W2 WHOLE — rank is ignored for it, and a small `factor`
            # makes each W2 nearly the full layer. Users reading "rank 6" expect a tiny
            # adapter, then the optimizer states OOM the GPU.
            print(
                f"  WARNING: adapter.full_matrix with factor={factor} produced a "
                f"{trainable / 1e6:.0f}M-parameter adapter (rank is ignored for the full W2). "
                "For a small LoKr set full_matrix = false, or use factor = -1 (balanced, "
                "tiny full matrices).",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Public API: configure, save, load
# ---------------------------------------------------------------------------

def configure(unet, text_encoder, text_encoder_2, adapter_config):
    """Apply LoKr to unet and both text encoders. Prefer LyCORIS if available."""
    rank = adapter_config["rank"]
    alpha = adapter_config["alpha"]
    factor = adapter_config.get("factor", -1)
    decompose_both = adapter_config.get("decompose_both", False)
    full_matrix = adapter_config.get("full_matrix", False)
    dtype = adapter_config.get("dtype")

    def freeze_and_apply_vendored(module, containers, prefix):
        for p in module.parameters():
            p.requires_grad_(False)
        _apply_lokr_vendored(module, containers, adapter_config, prefix)

    # Always vendored for SDXL: the LyCORIS backend hangs its adapter off the model root, so its
    # params fall outside the DeepSpeed pipeline layers (not placed on GPU, not trained). The
    # vendored path registers params on each nn.Linear instead.
    freeze_and_apply_vendored(unet, [unet.down_blocks, unet.mid_block, unet.up_blocks], "unet.")
    freeze_and_apply_vendored(text_encoder, [text_encoder], "text_encoder.")
    freeze_and_apply_vendored(text_encoder_2, [text_encoder_2], "text_encoder_2.")
    if is_main_process():
        print("LoKr (SDXL): using vendored backend")


def save(save_dir, state_dict, adapter_config):
    """Save LoKr state_dict in LyCORIS/Comfy format (adapter_model.safetensors with .alpha)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    sd = dict(state_dict)
    if LYCORIS_AVAILABLE:
        # State dict may come from pipeline; keys already have unet./text_encoder. prefix
        # LyCORIS/Comfy expect similar; add .alpha for each LoKr module
        lokr_modules = set()
        for k in list(sd.keys()):
            if ".lokr_w1" in k or ".lokr_w2" in k or "lokr_w1_a" in k or "lokr_w2_a" in k:
                base = k.rsplit(".", 1)[0]
                lokr_modules.add(base)
        alpha_val = adapter_config.get("alpha", adapter_config.get("rank"))
        for base in lokr_modules:
            sd[f"{base}.alpha"] = torch.tensor(float(alpha_val))
    else:
        lokr_modules = set()
        for k in list(sd.keys()):
            if "lokr_w1" in k or "lokr_w2" in k:
                base = k.rsplit(".", 1)[0]
                lokr_modules.add(base)
        alpha_val = adapter_config.get("alpha", adapter_config.get("rank"))
        for base in lokr_modules:
            sd[f"{base}.alpha"] = torch.tensor(float(alpha_val))
    atomic_save_safetensors(save_dir / "adapter_model.safetensors", sd)


def infer_lokr_config_from_state(state):
    """Infer minimal adapter_config from LoKr state dict (for load_and_fuse when adapter not configured)."""
    rank = 4
    for k, v in state.items():
        if ".lokr_w1_b" in k:
            rank = int(v.shape[0])
            break
        if ".lokr_w2_a" in k:
            rank = int(v.shape[1])
            break
    return {
        "type": "lokr",
        "rank": rank,
        "alpha": rank,
        "factor": -1,
        "decompose_both": False,
        "full_matrix": False,
        "dtype": next((v.dtype for v in state.values() if hasattr(v, "dtype")), torch.float32),
    }


def fuse(pipeline):
    """Fuse loaded LoKr weights into base weights (vendored only; LyCORIS raises)."""
    if LYCORIS_AVAILABLE:
        for module in (pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2):
            if getattr(module, "_lycoris_net", None) is not None:
                raise NotImplementedError(
                    "load_and_fuse_adapter for LoKr with LyCORIS backend is not implemented; "
                    "use vendored LoKr (uninstall lycoris-lora or use a env without it) to fuse."
                )
    for module in (pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2):
        _fuse_vendored_on_module(module)


def load(pipeline, adapter_path):
    """Load LoKr weights from adapter_path into pipeline (unet, text_encoder, text_encoder_2)."""
    adapter_path = Path(adapter_path)
    if is_main_process():
        print(f"Loading LoKr adapter weights from {adapter_path}")
    files = list(adapter_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {adapter_path}")
    state = safetensors.torch.load_file(files[0])

    modified = {}
    for k, v in state.items():
        key = re.sub(r"^(transformer|diffusion_model)\.", "", k)
        if key.endswith(".alpha"):
            continue
        modified[key] = v

    def subset_for(prefix, skip_longer_prefix=None):
        if skip_longer_prefix:
            return {k[len(prefix):]: v for k, v in modified.items() if k.startswith(prefix) and not k.startswith(skip_longer_prefix)}
        return {k[len(prefix):]: v for k, v in modified.items() if k.startswith(prefix)}

    if LYCORIS_AVAILABLE:
        for module in (pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2):
            net = getattr(module, "_lycoris_net", None)
            if net is not None:
                if module is pipeline.unet:
                    sub = subset_for("unet.")
                elif module is pipeline.text_encoder:
                    sub = subset_for("text_encoder.", "text_encoder_2.")
                else:
                    sub = subset_for("text_encoder_2.")
                if sub:
                    net.load_state_dict(sub, strict=False)
    else:
        for root, prefix, skip in [
            (pipeline.unet, "unet.", None),
            (pipeline.text_encoder, "text_encoder.", "text_encoder_2."),
            (pipeline.text_encoder_2, "text_encoder_2.", None),
        ]:
            sub = subset_for(prefix, skip) if skip else subset_for(prefix)
            if sub:
                root.load_state_dict(sub, strict=False)

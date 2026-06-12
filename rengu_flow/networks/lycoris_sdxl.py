"""LyCORIS library networks for SDXL — thin root-mapping over ``lycoris_attach``.

Same public surface as ``lokr_sdxl`` (configure/save/load/fuse). Exports use
kohya-flat names (``lora_unet_*`` / ``lora_te1_*`` / ``lora_te2_*``) so the files
load in ComfyUI/a1111 lycoris loaders.
"""

from pathlib import Path

import safetensors.torch

from rengu_flow.networks import lycoris_attach
from rengu_flow.networks.lycoris_meta import create_lycoris_kwargs
from rengu_flow.utils.common import is_main_process
from rengu_flow.utils.save_io import atomic_save_safetensors

PREFIX_MAP = {
    "unet.": "lora_unet_",
    "text_encoder.": "lora_te1_",
    "text_encoder_2.": "lora_te2_",
}


def _configure_roots(unet, text_encoder, text_encoder_2):
    # Same adapter targets as lora/lokr: the unet's down/mid/up blocks and both
    # text encoders in full.
    return [
        (unet, [unet.down_blocks, unet.mid_block, unet.up_blocks], "unet."),
        (text_encoder, [text_encoder], "text_encoder."),
        (text_encoder_2, [text_encoder_2], "text_encoder_2."),
    ]


def _load_roots(pipeline):
    return [
        (pipeline.unet, "lora_unet_"),
        (pipeline.text_encoder, "lora_te1_"),
        (pipeline.text_encoder_2, "lora_te2_"),
    ]


def configure(unet, text_encoder, text_encoder_2, adapter_config):
    """Attach the configured lycoris algorithm to the unet and both text encoders."""
    lycoris_attach.configure_roots(
        _configure_roots(unet, text_encoder, text_encoder_2), adapter_config
    )


def save(save_dir, state_dict, adapter_config):
    """Save the adapter snapshot as a kohya-flat lycoris file with metadata."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    flat = lycoris_attach.save_transform(state_dict, adapter_config, PREFIX_MAP)
    algo, _ = create_lycoris_kwargs(adapter_config)
    metadata = {
        "format": "pt",
        "rengu_adapter_type": str(adapter_config["type"]),
        "lycoris_algo": algo,
        "rank": str(adapter_config["rank"]),
    }
    atomic_save_safetensors(save_dir / "adapter_model.safetensors", flat, metadata)


def _read_state(adapter_path):
    adapter_path = Path(adapter_path)
    files = sorted(adapter_path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No .safetensors file found in {adapter_path}")
    return safetensors.torch.load_file(files[0])


def load(pipeline, adapter_path):
    """Load exported lycoris weights into the already-configured (attached) adapters."""
    if is_main_process():
        print(f"Loading LyCORIS adapter weights from {adapter_path}")
    lycoris_attach.load_into(_load_roots(pipeline), _read_state(adapter_path))


def fuse(pipeline):
    """Merge attached (configured + loaded) adapters into the base weights."""
    lycoris_attach.fuse_all(
        (pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2)
    )


def load_and_fuse(pipeline, adapter_path):
    """Merge an exported lycoris file straight into base weights (no configure)."""
    if is_main_process():
        print(f"Fusing LyCORIS adapter from {adapter_path}")
    lycoris_attach.fuse_weights_into(_load_roots(pipeline), _read_state(adapter_path))


def looks_like_lycoris_state(state) -> bool:
    """True for kohya-flat exports from this module (used only when no adapter type
    is configured; locon/dora files share the plain-LoRA format and may route to
    either loader)."""
    return any(
        k.startswith(("lora_unet_", "lora_te1_", "lora_te2_"))
        and (
            ".hada_" in k
            or ".lokr_" in k
            or ".oft_blocks" in k
            or k.endswith((".a1.weight", ".b1.weight", ".dora_scale"))
        )
        for k in state
    )

"""Contract (Protocol) and stub base class for training pipeline models.

The orchestrator expects any registered model to satisfy this interface.
Aligned with diffusion-pipe models/base.BasePipeline; see docs/developer/architecture.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

# Type aliases for the contract (avoid hard dependency on torch in this module for Phase 0).
# In practice: parameters are iterables of nn.Parameter; loss_fn is (output, label) -> loss tensor.
Parameters = Any
LossFn = Callable[..., Any]


def make_contiguous(*tensors: Any) -> tuple[Any, ...]:
    """Return tensors made contiguous (for pipeline data passing)."""
    out = []
    for x in tensors:
        if hasattr(x, "contiguous") and callable(getattr(x, "contiguous")):
            out.append(x.contiguous())
        else:
            out.append(x)
    return tuple(out)


@runtime_checkable
class ModelPipelineProtocol(Protocol):
    """Protocol that every pipeline model must implement for the orchestrator.

    Methods and attributes are used by the training loop (DeepSpeed pipeline,
    optimizer setup, checkpointing, eval). See REPORTE_EJECUTIVO §1.2 and
    diffusion-pipe models/base.BasePipeline.
    """

    # Class attributes (implementations define these on the class).
    framerate: float | None
    pixels_round_to_multiple: int
    checkpointable_layers: list[str]  # Layer type names for partition/checkpoint (e.g. ['TransformerLayer']).

    def load_diffusion_model(self) -> None:
        """Load the main diffusion/transformer model (lazy load if needed)."""
        ...

    def get_vae(self) -> Any:
        """Return the VAE module used for encoding/decoding latents."""
        ...

    def get_text_encoders(self) -> Any:
        """Return text encoder(s) used for conditioning."""
        ...

    def configure_adapter(self, adapter_config: dict[str, Any]) -> None:
        """Configure LoRA/LoKr or other adapter on the model from config (e.g. config['adapter'])."""
        ...

    def save_adapter(self, save_dir: str | Path, peft_state_dict: dict[str, Any]) -> None:
        """Save adapter weights to save_dir (e.g. safetensors)."""
        ...

    def load_adapter_weights(self, adapter_path: str | Path) -> None:
        """Load adapter weights from path into the model."""
        ...

    def load_and_fuse_adapter(self, path: str | Path) -> None:
        """Load adapter and fuse it into the base weights (inference-style)."""
        ...

    def save_model(self, save_dir: str | Path, diffusers_sd: dict[str, Any]) -> None:
        """Save full model state dict to save_dir (e.g. for full fine-tuned model)."""
        ...

    def get_preprocess_media_file_fn(self, augmentation_resolver: Any = None) -> Any:
        """Return a callable/object that preprocesses media (crop, resize, masks) per dataset spec."""
        ...

    def get_call_vae_fn(self, vae: Any) -> Callable[..., Any]:
        """Return a function that runs the VAE (encode/decode) for caching or forward."""
        ...

    def get_call_text_encoder_fn(self, text_encoder: Any) -> Callable[..., Any]:
        """Return a function that runs the text encoder for conditioning."""
        ...

    def prepare_inputs(self, inputs: Any, timestep_quantile: float | None = None) -> Any:
        """Prepare batch inputs for the pipeline (timesteps, latents, conditioning, etc.)."""
        ...

    def to_layers(self) -> Any:
        """Return a sequence of layers for ManualPipelineModule (DeepSpeed pipeline partition)."""
        ...

    def model_specific_dataset_config_validation(self, dataset_config: dict[str, Any]) -> None:
        """Optional validation of dataset TOML (resolutions, etc.) for this model."""
        ...

    def get_param_groups(self, parameters: Parameters) -> list[dict[str, Any]]:
        """Return param groups for the optimizer (e.g. [{'params': parameters}] or per-component)."""
        ...

    def get_loss_fn(self) -> LossFn:
        """Return a callable loss_fn(output, label) used by the pipeline (e.g. MSE with mask)."""
        ...

    def enable_block_swap(self, blocks_to_swap: Any) -> None:
        """Optional: enable block swap (offload blocks to CPU) for this model."""
        ...

    def prepare_block_swap_training(self) -> None:
        """Optional: set up block swap for training mode."""
        ...

    def prepare_block_swap_inference(self, disable_block_swap: bool = False) -> None:
        """Optional: set up block swap for inference/eval (e.g. disable for full model eval)."""
        ...

    def freeze_text_encoders(self) -> None:
        """Optional: freeze text encoder parameters (full-model mode, train UNet only). No-op by default."""
        ...


class BasePipeline:
    """Stub base class implementing ModelPipelineProtocol with NotImplementedError.

    Concrete models (Flux, SDXL, etc.) inherit and override. The orchestrator
    will only rely on the protocol; this class is for convenience and typing.
    """

    framerate: float | None = None
    pixels_round_to_multiple: int = 16
    checkpointable_layers: list[str] = []  # Subclasses override (e.g. ['TransformerLayer']).

    def load_diffusion_model(self) -> None:
        """Load the main diffusion/transformer model."""
        pass

    def get_vae(self) -> Any:
        """Return the VAE module."""
        raise NotImplementedError()

    def get_text_encoders(self) -> Any:
        """Return text encoder(s)."""
        raise NotImplementedError()

    def configure_adapter(self, adapter_config: dict[str, Any]) -> None:
        """Configure adapter from config."""
        raise NotImplementedError()

    def save_adapter(self, save_dir: str | Path, peft_state_dict: dict[str, Any]) -> None:
        """Save adapter weights."""
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path: str | Path) -> None:
        """Load adapter weights from path."""
        raise NotImplementedError()

    def load_and_fuse_adapter(self, path: str | Path) -> None:
        """Load and fuse adapter."""
        raise NotImplementedError()

    def save_model(self, save_dir: str | Path, diffusers_sd: dict[str, Any]) -> None:
        """Save full model."""
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self) -> Any:
        """Return preprocess callable."""
        raise NotImplementedError()

    def get_call_vae_fn(self, vae: Any) -> Callable[..., Any]:
        """Return VAE callable."""
        raise NotImplementedError()

    def get_call_text_encoder_fn(self, text_encoder: Any) -> Callable[..., Any]:
        """Return text encoder callable."""
        raise NotImplementedError()

    def prepare_inputs(self, inputs: Any, timestep_quantile: float | None = None) -> Any:
        """Prepare batch inputs."""
        raise NotImplementedError()

    def to_layers(self) -> Any:
        """Return sequence of layers for pipeline partition."""
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config: dict[str, Any]) -> None:
        """Optional dataset config validation."""
        pass

    def get_param_groups(self, parameters: Parameters) -> list[dict[str, Any]]:
        """Return param groups for optimizer (default: single group)."""
        return [{"params": parameters}]

    def get_loss_fn(self) -> LossFn:
        """Return loss callable (output, label) -> loss."""
        raise NotImplementedError()

    def enable_block_swap(self, blocks_to_swap: Any) -> None:
        """Enable block swap (optional)."""
        raise NotImplementedError("Block swapping is not implemented for this model")

    def prepare_block_swap_training(self) -> None:
        """Set up block swap for training."""
        pass

    def prepare_block_swap_inference(self, disable_block_swap: bool = False) -> None:
        """Set up block swap for inference."""
        pass

    def freeze_text_encoders(self) -> None:
        """Optional: freeze text encoder parameters for full-model UNet-only training. No-op if not supported."""
        pass

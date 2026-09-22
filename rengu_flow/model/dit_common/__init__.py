"""Building blocks shared by the flow-matching DiT pipelines (krea2, cosmos_predict2, ...).

Only code that is genuinely identical across models lives here; anything that needs a
per-model branch stays in the model package. See docs/developer/model-pipeline-contract.md.
"""

from rengu_flow.model.dit_common.flow_matching import (
    add_flow_noise,
    calculate_shift,
    sample_timesteps,
    shift_timesteps,
    time_shift,
)
from rengu_flow.model.dit_common.pipeline import DiTPipeline
from rengu_flow.model.dit_common.preview import preview_autocast, preview_compute_dtype
from rengu_flow.model.dit_common.quantize import quantize_frozen_dit
from rengu_flow.model.dit_common.text import compact_text_embeddings, pad_text_embeddings

__all__ = [
    "DiTPipeline",
    "add_flow_noise",
    "calculate_shift",
    "compact_text_embeddings",
    "pad_text_embeddings",
    "preview_autocast",
    "preview_compute_dtype",
    "quantize_frozen_dit",
    "sample_timesteps",
    "shift_timesteps",
    "time_shift",
]

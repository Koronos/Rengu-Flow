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

__all__ = [
    "DiTPipeline",
    "add_flow_noise",
    "calculate_shift",
    "preview_autocast",
    "preview_compute_dtype",
    "sample_timesteps",
    "shift_timesteps",
    "time_shift",
]

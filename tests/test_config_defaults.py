"""Tests for config defaults: set_config_defaults."""

import pytest
import torch

from rengu_flow.config.defaults import set_config_defaults
from rengu_flow.config.validation import ConfigValidationError


def test_set_config_defaults_top_level_and_model_dtype(minimal_config_copy):
    set_config_defaults(minimal_config_copy)
    assert minimal_config_copy.get("output_dir") == "output"
    assert minimal_config_copy.get("epochs") == 1
    assert minimal_config_copy.get("gradient_accumulation_steps") == 1
    assert minimal_config_copy.get("micro_batch_size_per_gpu") == 1
    assert minimal_config_copy.get("lr_scheduler") == "constant"
    assert minimal_config_copy.get("save_every_n_epochs") == 1
    assert "lr_scheduler_args" in minimal_config_copy
    dtype = minimal_config_copy["model"]["dtype"]
    assert dtype == torch.bfloat16 or dtype == "bfloat16"


@pytest.mark.parametrize("adapter_init, expected_rank, expected_alpha", [
    ({"type": "lora", "rank": 16}, 16, 16),
    ({"type": "lora", "dim": 8}, 8, 8),
], ids=["lora_rank", "lora_dim_alias"])
def test_set_config_defaults_adapter_lora(minimal_config_copy, adapter_init, expected_rank, expected_alpha):
    minimal_config_copy["adapter"] = adapter_init
    set_config_defaults(minimal_config_copy)
    adapter = minimal_config_copy["adapter"]
    assert adapter["rank"] == expected_rank
    assert adapter["alpha"] == expected_alpha
    assert adapter["dropout"] == 0.0
    assert "dtype" in adapter


@pytest.mark.parametrize("adapter_init, expected_rank", [
    ({"type": "lokr", "rank": 16}, 16),
    ({"type": "lokr", "dim": 4}, 4),
], ids=["lokr_rank", "lokr_dim_alias"])
def test_set_config_defaults_adapter_lokr(minimal_config_copy, adapter_init, expected_rank):
    minimal_config_copy["adapter"] = adapter_init
    set_config_defaults(minimal_config_copy)
    adapter = minimal_config_copy["adapter"]
    assert adapter["rank"] == expected_rank
    assert adapter["alpha"] == expected_rank
    assert adapter["factor"] == -1
    assert adapter["decompose_both"] is False
    assert adapter["full_matrix"] is False
    assert "dtype" in adapter


def test_set_config_defaults_rejects_explicit_alpha(minimal_config_copy):
    minimal_config_copy["adapter"] = {"type": "lora", "rank": 8, "alpha": 16}
    with pytest.raises(ConfigValidationError, match="alpha"):
        set_config_defaults(minimal_config_copy)


def test_set_config_defaults_adapter_unknown_raises(minimal_config_copy):
    minimal_config_copy["adapter"] = {"type": "other", "rank": 8}
    with pytest.raises(NotImplementedError) as exc_info:
        set_config_defaults(minimal_config_copy)
    assert "other" in str(exc_info.value)


def test_set_config_defaults_eval_and_monitoring(minimal_config_copy):
    """§1.6: eval and monitoring defaults (eval_before_first_step, disable_block_swap_for_eval, monitoring)."""
    set_config_defaults(minimal_config_copy)
    assert minimal_config_copy.get("eval_before_first_step") is True
    assert minimal_config_copy.get("disable_block_swap_for_eval") is False
    assert minimal_config_copy.get("steps_per_print") == 1
    assert minimal_config_copy.get("x_axis_examples") is False
    assert "monitoring" in minimal_config_copy
    mon = minimal_config_copy["monitoring"]
    assert mon.get("enable_wandb") is False
    assert mon.get("enable_status_file") is False
    assert "wandb_tracker_name" in mon
    assert mon.get("wandb_tracker_name") == "rengu-flow"
    assert "wandb_run_name" in mon


@pytest.mark.parametrize("legacy", ["selective", "unsloth"])
def test_set_config_defaults_degrades_retired_ac_modes(minimal_config_copy, legacy, capsys):
    """Retired AC modes (SAC/unsloth) fall back to full checkpointing with a warning."""
    minimal_config_copy["activation_checkpointing"] = legacy
    set_config_defaults(minimal_config_copy)
    assert minimal_config_copy["activation_checkpointing"] is True
    out = capsys.readouterr().out
    assert "retired" in out and "auto" in out

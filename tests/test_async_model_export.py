"""CPU tests for async export snapshot helper."""

from unittest.mock import patch

import torch

from renga_flow.utils.async_model_export import (
    async_snapshot_fits_in_ram,
    clone_state_dict_to_cpu,
    estimate_state_dict_bytes,
    format_byte_size,
)


def test_clone_state_dict_to_cpu_copies():
    w = torch.ones(2, 3)
    out = clone_state_dict_to_cpu({"a": w})
    assert out["a"].device.type == "cpu"
    w.add_(1.0)
    assert out["a"].sum().item() == 6.0


def test_clone_state_dict_to_cpu_dtype():
    w = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    out = clone_state_dict_to_cpu({"a": w}, save_dtype=torch.bfloat16)
    assert out["a"].dtype == torch.bfloat16


def test_estimate_state_dict_bytes_respects_save_dtype():
    w = torch.ones(100, dtype=torch.float32)
    assert estimate_state_dict_bytes({"a": w}) == 400
    assert estimate_state_dict_bytes({"a": w}, save_dtype=torch.bfloat16) == 200


def test_async_snapshot_fits_when_enough_ram():
    w = torch.ones(10)
    with patch(
        "renga_flow.utils.async_model_export._available_ram_bytes",
        return_value=10_000,
    ):
        fits, needed, available = async_snapshot_fits_in_ram({"a": w}, None)
    assert fits is True
    assert needed == 40
    assert available == 10_000


def test_async_snapshot_skips_when_not_enough_ram():
    w = torch.ones(100)
    with patch(
        "renga_flow.utils.async_model_export._available_ram_bytes",
        return_value=10,
    ):
        fits, needed, available = async_snapshot_fits_in_ram({"a": w}, None)
    assert fits is False
    assert needed == 400
    assert available == 10


def test_async_snapshot_max_snapshot_bytes():
    w = torch.ones(50)
    with patch(
        "renga_flow.utils.async_model_export._available_ram_bytes",
        return_value=10_000,
    ):
        fits, needed, _ = async_snapshot_fits_in_ram(
            {"a": w},
            None,
            max_snapshot_bytes=100,
        )
    assert fits is False
    assert needed == 200


def test_format_byte_size():
    assert format_byte_size(512) == "512 B"
    assert format_byte_size(1024 * 1024) == "1.0 MiB"
    assert format_byte_size(3 * 1024**3).endswith("GiB")

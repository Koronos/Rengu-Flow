"""Async export must never overlap two disk writes, and saves must wait for the in-flight one.

These guard the invariants the design relies on:
- one model is written to disk at a time (a second save waits for the first to finish, then runs);
- ``save_model`` / ``save_checkpoint`` block on any in-flight async write before touching disk;
- when a snapshot does not fit in RAM, the export falls back to a synchronous write.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import torch

from rengu_flow.utils.async_model_export import AsyncModelExportWriter, ModelExportJob
from rengu_flow.utils.saver import Saver


def _job(tmp_path, name: str) -> ModelExportJob:
    return ModelExportJob(
        name=name,
        save_dir=tmp_path / name,
        state_dict={},
        is_adapter=True,
        config_path=str(tmp_path / "config.toml"),
    )


def test_writer_never_overlaps_and_second_waits_for_first(tmp_path):
    """A slow write_fn proves submit() serializes: max concurrency 1, strict A-before-B order."""
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()
    order: list[tuple[str, str]] = []

    def slow_write(job: ModelExportJob) -> None:
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        order.append(("start", job.name))
        time.sleep(0.2)
        order.append(("end", job.name))
        with lock:
            state["active"] -= 1

    writer = AsyncModelExportWriter(slow_write)
    try:
        t0 = time.perf_counter()
        writer.submit(_job(tmp_path, "a"))  # returns immediately (writer was idle)
        # submit(b) must BLOCK until "a" finished writing, then enqueue "b".
        writer.submit(_job(tmp_path, "b"))
        waited = time.perf_counter() - t0
    finally:
        writer.shutdown()  # drains "b"

    # Two writes never ran at the same time.
    assert state["max_active"] == 1
    # Strict ordering: a fully completes before b begins.
    assert order == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]
    # submit(b) actually blocked on a's write (~0.2s) rather than returning instantly.
    assert waited >= 0.2


def test_writer_reraises_write_failure_on_next_submit(tmp_path):
    """A failed background write surfaces to the caller (not silently swallowed)."""

    def boom(_job: ModelExportJob) -> None:
        raise RuntimeError("disk exploded")

    writer = AsyncModelExportWriter(boom)
    writer.submit(_job(tmp_path, "a"))
    raised = False
    try:
        writer.submit(_job(tmp_path, "b"))  # wait_done() inside sees the stored error
    except RuntimeError:
        raised = True
    finally:
        try:
            writer.shutdown()
        except RuntimeError:
            pass
    assert raised


def _saver_with_mock_writer(tmp_path):
    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    model_engine = MagicMock()
    model_engine.grid.get_data_parallel_rank.return_value = 0
    model_engine.grid.get_pipe_parallel_rank.return_value = 0
    saver = Saver(
        args, {}, True, tmp_path, MagicMock(), MagicMock(), model_engine, MagicMock()
    )
    saver._async_writer = MagicMock()  # pretend async export is active
    return saver


def test_save_model_waits_for_inflight_export_before_writing(tmp_path):
    saver = _saver_with_mock_writer(tmp_path)
    seq: list[str] = []
    saver._async_writer.wait_done.side_effect = lambda: seq.append("wait")
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            with patch.object(
                saver, "_save_model_once", side_effect=lambda name: seq.append("save")
            ):
                saver.save_model("step1")
    # The in-flight async write is awaited BEFORE this save touches the disk.
    assert seq == ["wait", "save"]


def test_save_checkpoint_waits_for_inflight_export_before_writing(tmp_path):
    saver = _saver_with_mock_writer(tmp_path)
    seq: list[str] = []
    saver._async_writer.wait_done.side_effect = lambda: seq.append("wait")
    saver.model_engine.save_checkpoint.side_effect = lambda *a, **k: seq.append("ckpt")
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            with patch("rengu_flow.utils.saver.snapshot_global_step_dirs", return_value=set()):
                with patch("rengu_flow.utils.saver._prune_old_checkpoints"):
                    saver.save_checkpoint(1, 0)
    assert seq == ["wait", "ckpt"]


def test_snapshot_falls_back_to_sync_when_ram_short(tmp_path):
    """When the CPU snapshot would not fit in RAM, the async path defers to a synchronous write."""
    saver = _saver_with_mock_writer(tmp_path)
    saver.model_engine.grid.get_pipe_parallel_world_size.return_value = 1
    saver.pipeline_model.named_parameters.return_value = []
    saver.pipeline_model.parameters.return_value = []
    called = {"sync": False}
    with patch("rengu_flow.utils.saver.dist") as mock_dist:
        mock_dist.barrier = MagicMock()
        mock_dist.broadcast = MagicMock()
        with patch("rengu_flow.utils.saver.is_main_process", return_value=True):
            with patch(
                "rengu_flow.utils.saver.async_snapshot_fits_from_config",
                return_value=(False, 10**12, 1024),
            ):
                with patch.object(
                    saver,
                    "_run_pipeline_export_sync",
                    side_effect=lambda name, adapter_only: called.__setitem__("sync", True),
                ):
                    saver._run_pipeline_export_async("step1", adapter_only=True)
    assert called["sync"] is True
    saver._async_writer.submit.assert_not_called()


def test_estimate_accounts_for_inflight_snapshot_ram():
    """Sanity: a second snapshot is sized while the first still occupies RAM (psutil-driven)."""
    w = torch.ones(1000)
    # available reported AFTER the first snapshot already consumed RAM -> small -> does not fit.
    with patch(
        "rengu_flow.utils.async_model_export._available_ram_bytes", return_value=100
    ):
        from rengu_flow.utils.async_model_export import async_snapshot_fits_in_ram

        fits, _, _ = async_snapshot_fits_in_ram({"a": w}, None)
    assert fits is False

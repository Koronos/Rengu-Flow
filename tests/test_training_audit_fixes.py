"""Regression tests for the training-workflow audit (CLI, saver, async export, EMA, validation).

Each test failed against the code before the audit fix and passes after it.
"""

from __future__ import annotations

import errno
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

pytestmark = pytest.mark.no_ui_db


# ---------------------------------------------------------------- 1. CLI resume flag


def _launcher_cmd(monkeypatch, tmp_path, **kwargs):
    from rengu_flow.cli.train_launcher import build_train_command
    from rengu_flow.config.local_config import LocalConfig, TrainingConfig

    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    monkeypatch.setattr(
        "rengu_flow.cli.train_launcher.ensure_local_config_loaded",
        lambda: LocalConfig(root=tmp_path, training=TrainingConfig(num_gpus=1, master_port=29500)),
    )
    return build_train_command(tmp_path / "t.toml", **kwargs)


def test_bare_resume_flag_is_forwarded_without_value(monkeypatch, tmp_path):
    cmd = _launcher_cmd(monkeypatch, tmp_path, resume_from=True)
    assert "--resume_from_checkpoint" in cmd
    idx = cmd.index("--resume_from_checkpoint")
    assert idx == len(cmd) - 1 or cmd[idx + 1].startswith("-")


def test_resume_flag_with_value_and_without(monkeypatch, tmp_path):
    cmd = _launcher_cmd(monkeypatch, tmp_path, resume_from="run_a")
    assert cmd[cmd.index("--resume_from_checkpoint") + 1] == "run_a"
    assert "--resume_from_checkpoint" not in _launcher_cmd(monkeypatch, tmp_path, resume_from=None)


def test_run_train_passes_bare_resume_through(monkeypatch, tmp_path):
    import argparse

    from rengu_flow.cli import train_cmd
    from rengu_flow.config.local_config import LocalConfig, TrainingConfig

    cfg = LocalConfig(root=tmp_path, training=TrainingConfig(num_gpus=1, master_port=29500))
    monkeypatch.setattr(train_cmd, "ensure_local_config_loaded", lambda: cfg)
    monkeypatch.setattr("rengu_flow.cli.train_launcher.ensure_local_config_loaded", lambda: cfg)
    monkeypatch.setattr(train_cmd, "ensure_training_extras", lambda *a, **k: None)
    seen = {}
    monkeypatch.setattr(
        train_cmd, "run_training_with_progress", lambda cmd, env=None, cwd=None: (seen.update(cmd=cmd), 0)[1]
    )
    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    args = argparse.Namespace(
        config=str(tmp_path / "t.toml"), engine=None, num_gpus=None, master_port=None,
        resume_from_checkpoint=True, extra=[],
    )
    with pytest.raises(SystemExit):
        train_cmd.run_train(args)
    assert "--resume_from_checkpoint" in seen["cmd"]


# ---------------------------------------------------------------- 17. signal unlink: see test_signal_files


# ---------------------------------------------------------------- distributed.any_rank (12)


def test_any_rank_single_process_is_identity():
    from rengu_flow import distributed as dist

    assert dist.any_rank(True) is True
    assert dist.any_rank(False) is False


def test_any_rank_all_reduces_max_when_initialized(monkeypatch):
    from rengu_flow import distributed as dist

    comm = MagicMock()
    comm.ReduceOp.MAX = "MAX"

    def fake_all_reduce(t, op=None):
        assert op == "MAX"
        t.fill_(1)  # another rank reported an OOM

    comm.all_reduce.side_effect = fake_all_reduce
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "_comm", lambda: comm)
    assert dist.any_rank(False) is True  # this rank was fine, a peer was not


# ---------------------------------------------------------------- 5. async export shutdown


def _job(tmp_path, name):
    from rengu_flow.utils.async_model_export import ModelExportJob

    return ModelExportJob(name=name, save_dir=tmp_path / name, state_dict={}, is_adapter=True, config_path="c")


def test_shutdown_stops_worker_even_if_previous_write_failed(tmp_path):
    from rengu_flow.utils.async_model_export import AsyncModelExportWriter

    def boom(_job):
        raise OSError(errno.ENOSPC, "No space left on device")

    writer = AsyncModelExportWriter(boom)
    writer.submit(_job(tmp_path, "a"))
    try:
        writer.shutdown(timeout=5)
    except RuntimeError:
        pass  # the failure is still reported...
    assert not writer._worker.is_alive()  # ...but the worker is gone (process can exit)


def test_shutdown_drains_inflight_export_then_stops(tmp_path):
    from rengu_flow.utils.async_model_export import AsyncModelExportWriter

    done = threading.Event()

    def slow(_job):
        time.sleep(0.3)
        done.set()

    writer = AsyncModelExportWriter(slow)
    writer.submit(_job(tmp_path, "a"))
    writer.shutdown(timeout=5)
    assert done.is_set() and not writer._worker.is_alive()


def test_shutdown_timeout_still_lets_worker_exit_afterwards(tmp_path):
    from rengu_flow.utils.async_model_export import AsyncModelExportWriter

    release = threading.Event()
    writer = AsyncModelExportWriter(lambda _j: release.wait(5))
    writer.submit(_job(tmp_path, "a"))
    with pytest.raises(TimeoutError):
        writer.shutdown(timeout=0.1)
    release.set()
    writer._worker.join(timeout=5)
    assert not writer._worker.is_alive()  # sentinel was queued: no hang once the write ends


def test_writer_failure_is_reported_once(tmp_path):
    from rengu_flow.utils.async_model_export import AsyncModelExportWriter

    calls = {"n": 0}

    def flaky(_job):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOSPC, "full")

    writer = AsyncModelExportWriter(flaky)
    writer.submit(_job(tmp_path, "a"))
    with pytest.raises(RuntimeError):
        writer.wait_done()
    writer.wait_done()  # recovered: no stale error
    writer.submit(_job(tmp_path, "b"))
    writer.shutdown(timeout=5)


# ---------------------------------------------------------------- saver


def _saver(tmp_path, config=None):
    from rengu_flow.utils.saver import Saver

    args = MagicMock()
    args.config = str(tmp_path / "config.toml")
    (tmp_path / "config.toml").write_text("# test")
    engine = MagicMock()
    engine.grid.get_data_parallel_rank.return_value = 0
    engine.grid.get_pipe_parallel_rank.return_value = 0
    return Saver(args, config or {}, True, tmp_path, MagicMock(), MagicMock(), engine, MagicMock())


def _enospc_runtime_error():
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise RuntimeError("Async model export failed") from inner
    except RuntimeError as exc:
        return exc


def test_async_disk_full_enters_disk_wait_instead_of_crashing(tmp_path):
    """7: the background writer's ENOSPC surfaces in _wait_async_export; save_model must route it
    through the disk-full recovery (wait, retry) like a synchronous failure."""
    from rengu_flow.utils.signal_files import ExportRecoveryAction

    saver = _saver(tmp_path)
    saver._async_writer = MagicMock()
    saver._async_writer.wait_done.side_effect = [_enospc_runtime_error(), None]
    with patch("rengu_flow.utils.saver.dist"), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ), patch.object(saver, "_save_model_once") as once, patch(
        "rengu_flow.utils.saver.wait_for_export_recovery",
        return_value=ExportRecoveryAction.CONTINUE,
    ) as waited:
        assert saver.save_model("step5") is True
    waited.assert_called_once()
    once.assert_called_once_with("step5")


def test_async_disk_full_makes_checkpoint_return_false(tmp_path):
    saver = _saver(tmp_path)
    saver._async_writer = MagicMock()
    saver._async_writer.wait_done.side_effect = _enospc_runtime_error()
    with patch("rengu_flow.utils.saver.dist"), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ):
        assert saver.save_checkpoint(3, 12) is False
    saver.model_engine.save_checkpoint.assert_not_called()


def test_async_non_disk_failure_still_raises(tmp_path):
    saver = _saver(tmp_path)
    saver._async_writer = MagicMock()
    saver._async_writer.wait_done.side_effect = RuntimeError("serialization bug")
    with patch("rengu_flow.utils.saver.dist"), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ):
        with pytest.raises(RuntimeError, match="serialization bug"):
            saver.save_checkpoint(3, 12)


def test_save_quit_with_failed_checkpoint_exits_nonzero(tmp_path, capsys):
    from rengu_flow.utils.signal_files import SignalResult

    saver = _saver(tmp_path)
    signals = SignalResult(
        should_checkpoint=True, should_quit=True, should_export_model=False,
        should_export_quit=False, should_preview=False, should_reload_config=False,
    )
    with patch("rengu_flow.utils.saver.process_signals", return_value=signals), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ), patch.object(saver, "save_checkpoint", return_value=False):
        with pytest.raises(SystemExit) as exc:
            saver.process_step(10, 40)
    assert exc.value.code == 1
    assert "could NOT be written" in capsys.readouterr().out


def test_save_quit_with_successful_checkpoint_exits_zero(tmp_path):
    from rengu_flow.utils.signal_files import SignalResult

    saver = _saver(tmp_path)
    signals = SignalResult(
        should_checkpoint=True, should_quit=True, should_export_model=False,
        should_export_quit=False, should_preview=False, should_reload_config=False,
    )
    with patch("rengu_flow.utils.saver.process_signals", return_value=signals), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ), patch.object(saver, "save_checkpoint", return_value=True):
        with pytest.raises(SystemExit) as exc:
            saver.process_step(10, 40)
    assert exc.value.code == 0


def test_shutdown_async_exports_without_barrier_and_with_timeout(tmp_path):
    saver = _saver(tmp_path)
    saver._async_writer = MagicMock()
    with patch("rengu_flow.utils.saver.dist") as mock_dist, patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ):
        saver.shutdown_async_exports(timeout=7.0, sync_ranks=False)
    saver._async_writer.shutdown.assert_called_once_with(timeout=7.0)
    mock_dist.barrier.assert_not_called()


# ---------------------------------------------------------------- 8. checkpoint retention


def _mk_ckpts(root: Path, steps, latest):
    for s in steps:
        (root / f"global_step{s}").mkdir()
    (root / "latest").write_text(f"global_step{latest}")


def test_retention_never_deletes_latest_after_resume_from_old_tag(tmp_path, capsys):
    from rengu_flow.utils.saver import _prune_old_checkpoints

    # Resumed from step 100, just wrote step 150; steps 400/500/600 belong to the old branch.
    _mk_ckpts(tmp_path, [100, 150, 400, 500, 600], latest=150)
    _prune_old_checkpoints(tmp_path, 2)
    remaining = {p.name for p in tmp_path.iterdir() if p.is_dir()}
    assert "global_step150" in remaining  # what `latest` points at survives
    assert {"global_step400", "global_step500", "global_step600"} <= remaining  # not auto-deleted
    assert "global_step100" in remaining  # within max_keep=2 of the <= current ones
    assert "abandoned branch" in capsys.readouterr().out


def test_retention_prunes_oldest_among_not_ahead(tmp_path):
    from rengu_flow.utils.saver import _prune_old_checkpoints

    _mk_ckpts(tmp_path, [100, 200, 300, 400], latest=400)
    _prune_old_checkpoints(tmp_path, 2)
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {"global_step300", "global_step400"}


# ---------------------------------------------------------------- 9. EMA


def test_ema_reseed_from_live_weights():
    from rengu_flow.training.ema import TrainingEMA

    p = torch.nn.Parameter(torch.tensor([1.0]))
    ema = TrainingEMA([p], decay=0.9)
    with torch.no_grad():
        p.fill_(8.0)
    ema.reseed([p])
    assert ema.shadow[id(p)].item() == 8.0


def test_load_ema_checkpoint_missing_file_reseeds(tmp_path):
    from rengu_flow.training.ema import TrainingEMA, load_ema_checkpoint

    p = torch.nn.Parameter(torch.tensor([1.0]))
    ema = TrainingEMA([p], decay=0.9)
    with torch.no_grad():
        p.fill_(8.0)  # the checkpoint load replaced the weights after the EMA was built
    ckpt = tmp_path / "global_step9"
    ckpt.mkdir()
    load_ema_checkpoint(ckpt, ema, [p])
    assert ema.shadow[id(p)].item() == 8.0


def test_fork_checkpoint_saves_ema_under_fork_tag(tmp_path):
    from rengu_flow.training.ema import TrainingEMA
    from rengu_flow.utils.saver import FORK_CHECKPOINT_TAG

    p = torch.nn.Parameter(torch.tensor([2.0]))
    saver = _saver(tmp_path)
    saver.training_ema = TrainingEMA([p], decay=0.9)
    saver.pipeline_model.parameters.return_value = [p]
    # The engine "writes" the fork dir; `latest` is NOT updated (save_latest=False).
    saver.model_engine.save_checkpoint.side_effect = lambda *a, **k: (
        tmp_path / k["tag"]
    ).mkdir()
    (tmp_path / "global_step1").mkdir()
    (tmp_path / "latest").write_text("global_step1")
    with patch("rengu_flow.utils.saver.dist"), patch(
        "rengu_flow.utils.saver.is_main_process", return_value=True
    ), patch("rengu_flow.training.ema.torch.save") as save:
        assert saver.save_fork_checkpoint(180, 720) is True
    dest = save.call_args[0][1]
    assert Path(dest) == tmp_path / FORK_CHECKPOINT_TAG / "ema.pt"


# ---------------------------------------------------------------- 14/15. validation


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 0),
        ("max_steps", 0),
        ("gradient_accumulation_steps", 0),
        ("save_every_n_steps", 0),
        ("save_every_n_epochs", 0),
        ("eval_every_n_steps", 0),
        ("eval_every_n_epochs", 0),
        ("checkpoint_every_n_epochs", 0),
    ],
)
def test_cadence_below_one_is_rejected(key, value):
    from rengu_flow.config.validation import collect_cadence_issues

    issues = collect_cadence_issues({key: value})
    assert any(key in i and ">= 1" in i for i in issues)


@pytest.mark.parametrize("key", ["save_every_n_examples", "eval_every_n_examples"])
def test_examples_cadence_smaller_than_a_step_is_rejected(key):
    from rengu_flow.config.validation import collect_cadence_issues

    config = {"micro_batch_size_per_gpu": 2, "gradient_accumulation_steps": 4, key: 5}
    assert any(key in i and "0 steps" in i for i in collect_cadence_issues(config))
    config[key] = 8
    assert collect_cadence_issues(config) == []


def test_cadence_issues_reach_collect_validation_errors():
    from rengu_flow.config.validation import collect_validation_errors

    issues = collect_validation_errors({"epochs": 0, "save_every_n_steps": 0})
    assert any("epochs must be" in i for i in issues)
    assert any("save_every_n_steps must be" in i for i in issues)


@pytest.mark.parametrize(
    "toml_text,needle",
    [
        ('dataset = "d.toml"\n[model]\ntype = "sdxl"\ndtype = "bf16"\n[optimizer]\ntype = "adamw"\n', "model.dtype"),
        ('dataset = "d.toml"\n[optimizer]\ntype = "adamw"\n', "[model]"),
        (
            'dataset = "d.toml"\n[model]\ntype = "sdxl"\ndtype = "bfloat16"\n'
            '[adapter]\ntype = "lora"\n[optimizer]\ntype = "adamw"\n',
            "adapter.rank",
        ),
    ],
)
def test_validate_only_reports_friendly_error_not_keyerror(tmp_path, toml_text, needle):
    from rengu_flow.main import parse_args, run_prepared

    cfg = tmp_path / "t.toml"
    cfg.write_text(toml_text, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_prepared(parse_args(["--config", str(cfg), "--validate-only"]))
    assert "Config validation failed" in str(exc.value)
    assert needle in str(exc.value)


def test_validate_only_runs_backend_validation(tmp_path, monkeypatch):
    """validate must reject what train rejects (engine capability checks)."""
    from rengu_flow.main import parse_args, run_prepared

    monkeypatch.setenv("RENGU_ENGINE", "accelerate")
    cfg = tmp_path / "t.toml"
    cfg.write_text(
        'dataset = "d.toml"\npipeline_stages = 2\n[model]\ntype = "sdxl"\ndtype = "bfloat16"\n'
        'checkpoint_path = "x.safetensors"\n[optimizer]\ntype = "adamw"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_prepared(parse_args(["--config", str(cfg), "--validate-only"]))
    assert "pipeline_stages > 1 requires engine='deepspeed'" in str(exc.value)


# ---------------------------------------------------------------- 19. gradient_release param groups


def test_gradient_release_applies_weight_decay_split_and_group_options(monkeypatch):
    import sys

    from rengu_flow.main import _build_optimizer

    monkeypatch.setitem(sys.modules, "deepspeed", MagicMock())
    w2d = torch.nn.Parameter(torch.ones(2, 2))
    b1d = torch.nn.Parameter(torch.ones(2))

    model = MagicMock()
    model.get_param_groups.return_value = [{"params": [w2d, b1d], "lr": 3e-4}]
    config = {"optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.1, "gradient_release": True}}
    opt = _build_optimizer(
        [w2d, b1d],
        config=config,
        model=model,
        pipeline_model=MagicMock(),
        ds_config={"gradient_accumulation_steps": 1},
        global_batch_size=1,
        gradient_release=True,
    )
    by_param = {id(g["params"][0]): g for o in opt.optimizers for g in o.param_groups}
    assert by_param[id(w2d)]["weight_decay"] == 0.1
    assert by_param[id(b1d)]["weight_decay"] == 0  # 1-D params never get weight decay
    assert by_param[id(w2d)]["lr"] == 3e-4 and by_param[id(b1d)]["lr"] == 3e-4

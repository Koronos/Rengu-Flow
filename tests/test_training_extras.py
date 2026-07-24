"""Auto-install optional extras from training config."""

from __future__ import annotations

from unittest.mock import MagicMock


# Logic is centralized in rengu_flow.install.manager; the cli.training_extras module is a thin
# re-export shim, so patch/inspect the manager (the real home) here.
from rengu_flow.install import manager as training_extras
from rengu_flow.registry.optimizers import OPTIMIZER_ALIASES, VENDOR_OPTIMIZER_ALIASES


def test_profiles_for_cosmos_config():
    data = {
        "model": {"type": "cosmos_predict2", "dtype": "bfloat16"},
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "dataset": "d.toml",
    }
    assert training_extras.profiles_for_config_dict(data) == ["cosmos"]


def test_profiles_for_sdxl_lokr_adds_lycoris():
    data = {
        "model": {"type": "sdxl"},
        "adapter": {"type": "lokr"},
        "optimizer": {"type": "adamw"},
    }
    assert training_extras.profiles_for_config_dict(data) == ["lycoris"]


def test_profiles_for_adamw8bit_adds_optim():
    data = {
        "model": {"type": "sdxl"},
        "optimizer": {"type": "adamw8bit", "lr": 1e-4},
    }
    assert training_extras.profiles_for_config_dict(data) == ["optim"]


def test_profiles_for_prodigy_adds_optim():
    data = {
        "model": {"type": "sdxl"},
        "optimizer": {"type": "prodigy", "lr": 1.0},
    }
    assert training_extras.profiles_for_config_dict(data) == ["optim"]


def test_kaon_optimizers_route_to_kaon_profile() -> None:
    """All kaon-backed aliases install the kaon profile."""
    kaon_names = [n for n, (mod, _c) in OPTIMIZER_ALIASES.items() if mod == "kaon"]
    assert {
        "adakaon",
        "adamuon",
        "kprodigy",
        "lion",
        "adapnm",
        "adabelief",
        "adamp",
        "adopt",
        "schedulefree",
        "lookahead",
        "sam",
    } <= set(kaon_names)
    for name in kaon_names:
        profiles = training_extras.profiles_for_config_dict({"optimizer": {"type": name}})
        assert "kaon" in profiles, f"{name!r} did not route to the kaon profile"


def test_registry_optional_optimizers_map_to_an_install_profile() -> None:
    """Every dropdown alias needing a non-base dependency must map to an install profile
    (the [optim] extra, or a git-backed profile like 'kaon')."""
    names = set(OPTIMIZER_ALIASES) | set(VENDOR_OPTIMIZER_ALIASES)
    for name in names:
        profiles = training_extras.profiles_for_config_dict({"optimizer": {"type": name}})
        assert profiles, f"optimizer alias {name!r} maps to no install profile"


def test_profiles_for_genericoptim_adds_optim():
    data = {
        "model": {"type": "cosmos_predict2"},
        "optimizer": {"type": "genericoptim", "lr": 1e-4},
    }
    profiles = training_extras.profiles_for_config_dict(data)
    assert "cosmos" in profiles
    assert "optim" in profiles


def test_ensure_training_extras_skips_when_installed(tmp_path, monkeypatch):
    cfg = tmp_path / "train.toml"
    cfg.write_text('[model]\ntype = "sdxl"\n', encoding="utf-8")
    monkeypatch.setattr(training_extras, "profiles_for_config_path", lambda _p: ["cosmos"])
    monkeypatch.setattr(training_extras, "missing_profiles", lambda _p: [])
    assert training_extras.ensure_training_extras(cfg) == []


def test_ensure_training_extras_runs_sync(tmp_path, monkeypatch):
    cfg = tmp_path / "train.toml"
    cfg.write_text('[model]\ntype = "cosmos_predict2"\n', encoding="utf-8")
    monkeypatch.setattr(training_extras, "missing_profiles", lambda p: list(p))
    monkeypatch.setattr(training_extras, "profile_installed", lambda _p: True)
    synced: list[list[str]] = []

    def fake_ensure(profiles, *, root=None, reason=""):
        synced.append(profiles)
        return profiles

    monkeypatch.setattr(training_extras, "ensure_profiles", fake_ensure)
    result = training_extras.ensure_training_extras(cfg, root=tmp_path)
    assert result == ["cosmos"]
    assert synced == [["cosmos"]]


def test_jobs_start_calls_ensure_training_extras(tmp_path, monkeypatch):
    from rengu_flow_ui import jobs

    called = []

    def fake_ensure(path, *, root=None):
        called.append(path)

    monkeypatch.setattr(jobs, "ensure_training_extras", fake_ensure)
    monkeypatch.setattr(
        jobs,
        "build_train_command",
        lambda *a, **k: ["echo", "train"],
    )
    monkeypatch.setattr(jobs.db, "update_job", lambda *a, **k: None)

    proc = MagicMock()
    proc.pid = 12345
    from rengu_flow_ui import subprocess_util

    monkeypatch.setattr(
        subprocess_util,
        "popen_repo_subprocess",
        lambda *a, **k: (proc, None),
    )

    job = jobs.db.JobRecord(
        id=1,
        config_path=str(tmp_path / "c.toml"),
        log_path=str(tmp_path / "log.txt"),
        state="pending",
        num_gpus=1,
        extra_args="",
        resume_from=None,
        run_dir=None,
        output_dir=None,
        pid=None,
        started_at="2020-01-01T00:00:00Z",
        finished_at=None,
        exit_code=None,
    )
    jobs.start_job(job)
    assert called == [tmp_path / "c.toml"]

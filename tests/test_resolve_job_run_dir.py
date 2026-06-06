"""resolve_job_run_dir must not borrow an unrelated/older run's folder.

Regression for: a fresh "new run from this config" showed the SOURCE run's stats
because the fallback returned the newest existing folder while the new run's own
folder was still being created.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace

from rengu_flow_ui import training_hub


def _mk_run(output: "os.PathLike[str] | str", name: str, mtime_epoch: float):
    from pathlib import Path

    d = Path(output) / name
    d.mkdir(parents=True, exist_ok=True)
    # A run folder must look like one to the scanner (needs a .toml or status/events).
    (d / "config.toml").write_text("epochs = 1\n", encoding="utf-8")
    os.utime(d, (mtime_epoch, mtime_epoch))
    return d


def _job(output_dir, started_epoch, *, run_name=None, state="running", run_dir=None):
    started_at = datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat()
    content = f'run_name = "{run_name}"\n' if run_name else "epochs = 1\n"
    return SimpleNamespace(
        run_dir=run_dir,
        state=state,
        output_dir=str(output_dir),
        started_at=started_at,
        config_content=content,
    )


def test_named_new_run_is_not_confused_with_source(tmp_path):
    # Date-first folders: "{timestamp}_{name}".
    src = _mk_run(tmp_path, "20260101_10-00-00_myrun", 1_000.0)
    new = _mk_run(tmp_path, "20260605_10-00-00_myrun_2", 1_800.0)
    job = _job(tmp_path, started_epoch=1_500.0, run_name="myrun_2")
    assert training_hub.resolve_job_run_dir(job) == new.resolve()
    assert training_hub.resolve_job_run_dir(job) != src.resolve()


def test_nameless_run_picks_folder_created_after_start_not_older_source(tmp_path):
    src = _mk_run(tmp_path, "20260101_10-00-00", 1_000.0)
    new = _mk_run(tmp_path, "20260605_10-00-00", 1_800.0)
    job = _job(tmp_path, started_epoch=1_500.0)  # no run_name
    assert training_hub.resolve_job_run_dir(job) == new.resolve()
    assert training_hub.resolve_job_run_dir(job) != src.resolve()


def test_returns_none_when_only_a_pre_existing_source_folder_exists(tmp_path):
    # New run's folder not created yet: must NOT borrow the older source folder.
    _mk_run(tmp_path, "20260101_10-00-00_myrun", 1_000.0)
    job = _job(tmp_path, started_epoch=1_500.0, run_name="myrun_2")
    assert training_hub.resolve_job_run_dir(job) is None


def test_recorded_run_dir_is_used_as_is(tmp_path):
    d = _mk_run(tmp_path, "whatever", 1_000.0)
    job = _job(tmp_path, started_epoch=1_500.0, run_dir=str(d))
    assert training_hub.resolve_job_run_dir(job) == d.resolve()


def test_terminal_job_without_run_dir_gets_no_fallback(tmp_path):
    _mk_run(tmp_path, "20260605_10-00-00", 1_800.0)
    job = _job(tmp_path, started_epoch=1_500.0, state="finished")
    assert training_hub.resolve_job_run_dir(job) is None

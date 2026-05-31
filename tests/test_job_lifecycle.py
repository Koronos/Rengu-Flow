"""RF-06 / RF-07: per-job log persistence, run_dir capture, and failed-vs-finished state."""

from pathlib import Path

from rengu_flow_ui import db, jobs


def _make_running_job(log: Path) -> db.JobRecord:
    job = db.create_job(
        config_path="x",
        config_id=None,
        log_path=str(log),
        num_gpus=1,
        output_dir="output",
    )
    # pid=None so poll_job treats the process as gone and reconciles the final state.
    db.update_job(job.id, state="running", pid=None)
    return db.get_job(job.id)


def test_update_job_persists_log_path(ui_data_tmp: Path) -> None:
    """RF-07 root cause: log_path was missing from update_job's allowed set, so the per-job
    log path was silently dropped and every job kept the shared pending.log."""
    job = db.create_job(config_path="", config_id=None, log_path="logs/pending.log", num_gpus=1)
    db.update_job(job.id, log_path="logs/42.log")
    assert db.get_job(job.id).log_path == "logs/42.log"


def test_poll_job_marks_failed_on_nonzero(ui_data_tmp: Path, tmp_path: Path) -> None:
    log = tmp_path / "fail.log"
    log.write_text("boom\n[x:341] foo exits with return code = 1\n", encoding="utf-8")
    job = _make_running_job(log)
    out = jobs.poll_job(job.id)
    assert out.state == "failed"
    assert out.exit_code == 1


def test_poll_job_finished_on_success(ui_data_tmp: Path, tmp_path: Path) -> None:
    log = tmp_path / "ok.log"
    log.write_text("step=4/4\n[launch.py:367:main] Process 1 exits successfully.\n", encoding="utf-8")
    job = _make_running_job(log)
    out = jobs.poll_job(job.id)
    assert out.state == "finished"
    assert out.exit_code == 0


def test_read_exit_code_detects_traceback(ui_data_tmp: Path, tmp_path: Path) -> None:
    log = tmp_path / "tb.log"
    log.write_text("Traceback (most recent call last):\nValueError: x\n", encoding="utf-8")
    job = _make_running_job(log)
    assert jobs._read_exit_code(db.get_job(job.id)) == 1


def test_poll_job_captures_run_dir(ui_data_tmp: Path, tmp_path: Path) -> None:
    """RF-07: each job must point at its own run_dir (parsed from its log), not the newest
    folder in output_dir."""
    rundir = tmp_path / "myrun_20260101"
    rundir.mkdir()
    log = tmp_path / "run.log"
    log.write_text(f"Run dir: {rundir}\nProcess 1 exits successfully.\n", encoding="utf-8")
    job = _make_running_job(log)
    out = jobs.poll_job(job.id)
    assert out.run_dir == str(rundir.resolve())
    assert out.state == "finished"

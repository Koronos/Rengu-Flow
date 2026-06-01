"""Exit-code and run-dir parsing must scope to the CURRENT run segment of an appended job log."""

from __future__ import annotations

from types import SimpleNamespace

from rengu_flow_ui import jobs


def test_exit_code_and_run_dir_use_current_run_segment(tmp_path) -> None:
    new_run_dir = tmp_path / "run_new"
    new_run_dir.mkdir()
    log = tmp_path / "7.log"
    # A previous FAILED run, then the current CLEAN run — both appended to the same log file.
    log.write_text(
        "--- rengu-flow-ui job 7 ---\n"
        "Run dir: /nonexistent/run_old\n"
        "[rank0]: Traceback (most recent call last):\n"
        "[ERROR] exits with return code = 1\n"
        "--- rengu-flow-ui job 7 ---\n"
        f"Run dir: {new_run_dir}\n"
        "steps: 1 loss: 0.1\n"
        "Manually quitting (save_quit)\n"
        "Process 123 exits successfully.\n",
        encoding="utf-8",
    )
    job = SimpleNamespace(id="7", log_path=str(log), run_dir="/nonexistent/run_old")

    # The old "return code = 1" / Traceback must NOT mark the current clean run as failed.
    assert jobs._read_exit_code(job) == 0
    # run_dir must resolve to the current run's folder, not the stale previous one.
    assert jobs._parse_run_dir_from_log(job) == str(new_run_dir.resolve())

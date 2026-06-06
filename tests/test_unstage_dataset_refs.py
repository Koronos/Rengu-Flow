"""Tests for reversing per-job staging dataset paths back to original references."""

from __future__ import annotations

import toml

from rengu_flow_ui import job_import


def _set_existing(monkeypatch, ids: set[int]) -> None:
    monkeypatch.setattr(
        "rengu_flow_ui.job_import.library_db.dataset_exists",
        lambda did: int(did) in ids,
    )


def test_staging_lib_path_reverts_to_library_ref(monkeypatch):
    _set_existing(monkeypatch, {3})
    content = 'dataset = "/data/.rengu-flow-ui/staging/1/3.dataset.toml"\n'
    out = job_import.unstage_config_dataset_refs(content)
    assert toml.loads(out)["dataset"] == "rengu-flow-dataset:3"


def test_staging_lib_path_deleted_dataset_falls_back_to_run_copy(monkeypatch, tmp_path):
    _set_existing(monkeypatch, set())  # dataset 3 no longer in the library
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "3.dataset.toml").write_text("resolutions = [512]\n", encoding="utf-8")
    content = 'dataset = "/data/.rengu-flow-ui/staging/1/3.dataset.toml"\n'
    out = job_import.unstage_config_dataset_refs(content, run_dir=run_dir)
    assert toml.loads(out)["dataset"] == str((run_dir / "3.dataset.toml").resolve())


def test_staging_lib_path_deleted_no_copy_is_unchanged(monkeypatch):
    _set_existing(monkeypatch, set())
    content = 'dataset = "/data/.rengu-flow-ui/staging/1/3.dataset.toml"\n'
    out = job_import.unstage_config_dataset_refs(content)
    assert toml.loads(out)["dataset"] == "/data/.rengu-flow-ui/staging/1/3.dataset.toml"


def test_merged_staging_path_uses_run_copy(monkeypatch, tmp_path):
    _set_existing(monkeypatch, {3})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_dataset_merged.toml").write_text("resolutions=[512]\n", encoding="utf-8")
    content = 'dataset = "/data/.rengu-flow-ui/staging/2/training_dataset_merged.toml"\n'
    out = job_import.unstage_config_dataset_refs(content, run_dir=run_dir)
    assert toml.loads(out)["dataset"] == str((run_dir / "training_dataset_merged.toml").resolve())


def test_non_staging_path_and_library_ref_unchanged(monkeypatch):
    _set_existing(monkeypatch, {3})
    # A plain absolute dataset path (not under staging) is left alone.
    content1 = 'dataset = "/home/me/datasets/my.toml"\n'
    assert job_import.unstage_config_dataset_refs(content1) == content1
    # A library ref is already canonical -> unchanged text.
    content2 = 'dataset = "rengu-flow-dataset:3:artist"\n'
    assert job_import.unstage_config_dataset_refs(content2) == content2


def test_list_form_reverts_each_entry(monkeypatch):
    _set_existing(monkeypatch, {3, 7})
    content = (
        'dataset = ['
        '"/data/.rengu-flow-ui/staging/1/3.dataset.toml", '
        '"/data/.rengu-flow-ui/staging/1/7.dataset.toml"]\n'
    )
    out = toml.loads(job_import.unstage_config_dataset_refs(content))
    assert out["dataset"] == ["rengu-flow-dataset:3", "rengu-flow-dataset:7"]


def test_clean_config_returned_verbatim(monkeypatch):
    _set_existing(monkeypatch, {3})
    content = 'dataset = "rengu-flow-dataset:3"\nepochs = 5\n# keep me\n'
    # No staging path -> identical text (comments/formatting preserved).
    assert job_import.unstage_config_dataset_refs(content) == content


def test_eval_datasets_entries_reverted(monkeypatch):
    _set_existing(monkeypatch, {5})
    content = (
        'dataset = "rengu-flow-dataset:5"\n'
        'eval_datasets = ["/data/.rengu-flow-ui/staging/9/5.dataset.toml"]\n'
    )
    out = toml.loads(job_import.unstage_config_dataset_refs(content))
    assert out["eval_datasets"] == ["rengu-flow-dataset:5"]

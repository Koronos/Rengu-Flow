"""rengu prep CLI: parsing, TOML/flag precedence, runner markers + signal stop."""

import argparse
import json
import shutil
from pathlib import Path

import pytest

from rengu_flow.cli import prep_cmd
from rengu_flow.prep.config import PrepConfig, load_prep_config, parse_prep_config
from rengu_flow.prep.runner import run_stage

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


def _parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    prep_cmd.add_parser(sub)
    return parser.parse_args(argv)


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("a.jpg", "b.jpg"):
        shutil.copy(FIXTURE_JPG, d / name)
    return d


def test_parse_tag_args():
    args = _parse(["prep", "tag", "--path", "/x", "--model", "pixai-v0.9", "--overwrite"])
    assert args.prep_stage == "tag"
    config = prep_cmd._build_config(args)
    assert config.path == "/x"
    assert config.tag.models == ["pixai-v0.9"]
    assert config.tag.overwrite is True


def test_flags_override_toml(tmp_path):
    toml_path = tmp_path / "prep.toml"
    toml_path.write_text(
        'path = "/from/toml"\ncaption_ext = ".cap"\n\n[caption]\nmodel = "toriigate-0.5"\nquantization = "nf4"\n'
    )
    args = _parse(
        ["prep", "caption", "--config", str(toml_path), "--path", "/from/flag", "--quant", "int8"]
    )
    config = prep_cmd._build_config(args)
    assert config.path == "/from/flag"  # flag wins
    assert config.caption_ext == ".cap"  # toml survives
    assert config.caption.model == "toriigate-0.5"
    assert config.caption.quantization == "int8"  # flag wins


def test_unknown_toml_keys_dropped_gracefully(tmp_path):
    toml_path = tmp_path / "prep.toml"
    toml_path.write_text('path = "/x"\nbogus_key = 1\n\n[tag]\nancient_option = true\n')
    config = load_prep_config(toml_path)
    assert config.path == "/x"
    assert not hasattr(config.tag, "ancient_option")


def test_validate_rejects_bad_stage_and_missing_path(img_dir):
    config = parse_prep_config({"path": str(img_dir)})
    with pytest.raises(ValueError):
        config.validate_for_stage("explode")
    config = PrepConfig()
    with pytest.raises(ValueError):
        config.validate_for_stage("tag")


def test_run_stage_tag_writes_line1_report_and_markers(img_dir, tmp_path, capsys, monkeypatch):
    job_dir = tmp_path / "job"
    (img_dir / "a.txt").write_text("existing tags\n")  # skipped (overwrite=False)

    def fake_run_ensemble(paths, specs, **kwargs):
        on_progress = kwargs.get("on_progress")
        if on_progress:
            on_progress(len(paths), len(paths), "model fake")
        return {str(p): "1girl, solo" for p in paths}

    monkeypatch.setattr("rengu_flow.prep.tagger.run_ensemble", fake_run_ensemble)
    config = parse_prep_config({"path": str(img_dir)})
    code = run_stage(config, "tag", job_dir)
    assert code == 0

    assert (img_dir / "b.txt").read_text() == "1girl, solo\n"
    assert (img_dir / "a.txt").read_text() == "existing tags\n"  # untouched

    report = json.loads((job_dir / "report.json").read_text())
    assert report["tagged"] == 1 and report["skipped"] == 1
    out = capsys.readouterr().out
    assert "@@RFPROG@@" in out
    assert "prep tag exits with return code = 0" in out


def test_run_stage_failure_writes_error_report(img_dir, tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("rengu_flow.prep.tagger.run_ensemble", boom)
    config = parse_prep_config({"path": str(img_dir)})
    code = run_stage(config, "tag", tmp_path / "job")
    assert code == 1
    report = json.loads((tmp_path / "job" / "report.json").read_text())
    assert "model exploded" in report["error"]
    assert "prep tag exits with return code = 1" in capsys.readouterr().out


def test_run_stage_signal_file_stops_gracefully(img_dir, tmp_path, monkeypatch):
    from rengu_flow.utils.signal_files import SIGNAL_SAVE_QUIT

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / SIGNAL_SAVE_QUIT).touch()  # stop before the first batch

    calls = []

    def fake_run_ensemble(paths, specs, **kwargs):
        should_stop = kwargs.get("should_stop")
        assert should_stop is not None and should_stop()
        calls.append(len(paths))
        return {}

    monkeypatch.setattr("rengu_flow.prep.tagger.run_ensemble", fake_run_ensemble)
    config = parse_prep_config({"path": str(img_dir)})
    code = run_stage(config, "tag", job_dir)
    assert code == 0
    report = json.loads((job_dir / "report.json").read_text())
    assert report["stopped"] is True


def test_tagger_should_stop_breaks_between_batches(img_dir):
    from rengu_flow.prep.tagger import KNOWN_TAGGERS, run_ensemble

    stop_after = {"n": 1}
    seen_batches = []

    def factory(spec):
        def infer(paths):
            seen_batches.append(list(paths))
            return [{"tag": 0.9} for _ in paths]

        return infer

    def should_stop():
        return len(seen_batches) >= stop_after["n"]

    paths = sorted(img_dir.glob("*.jpg"))
    result = run_ensemble(
        paths,
        [KNOWN_TAGGERS["pixai-v0.9"]],
        batch_size=1,
        should_stop=should_stop,
        infer_factory=factory,
    )
    assert len(seen_batches) == 1  # stopped before the second batch
    assert len(result) == 1

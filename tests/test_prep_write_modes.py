"""Prep write modes (skip / replace / append on any 1-based line) and caption layout conversion.

Real ``CaptionStore`` IO throughout; the only doubles are the model (a stub tagger / backend), so
the sidecar <-> ``captions.json`` paths are exercised exactly as a workflow node runs them.
"""

import json
import shutil
from pathlib import Path

import pytest

from rengu_flow.prep.caption_store import (
    CaptionStore,
    convert_captions,
    effective_write_mode,
    merge_tags,
)
from rengu_flow.prep.captioner import CaptionBackend, CaptionerConfig, caption_folder
from rengu_flow.prep.config import PrepConfig, parse_prep_config
from rengu_flow.prep.runner import run_stage

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


@pytest.fixture(autouse=True)
def _isolate_prep_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(tmp_path / "appdata"))


@pytest.fixture
def img_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        shutil.copy(FIXTURE_JPG, d / name)
    return d


def _open(folder, fmt="sidecar", ext=".txt"):
    return CaptionStore.open(folder, fmt=fmt, ext=ext, positional=True)


# ------------------------------------------------------------------ write_line


@pytest.mark.parametrize("fmt", ["sidecar", "json"])
def test_target_line_3_on_empty_caption_pads_lines_1_and_2(img_dir, fmt):
    cs = _open(img_dir, fmt)
    cs.write_line("a.jpg", 2, "third line")
    assert cs.get_lines("a.jpg") == ["", "", "third line"]
    cs.save()
    # The padding survives a reload, in both layouts.
    assert _open(img_dir, fmt).get_lines("a.jpg") == ["", "", "third line"]
    if fmt == "sidecar":
        assert (img_dir / "a.txt").read_text() == "\n\nthird line\n"
    else:
        assert json.loads((img_dir / "captions.json").read_text())["a.jpg"] == [
            "",
            "",
            "third line",
        ]


@pytest.mark.parametrize("fmt", ["sidecar", "json"])
def test_replace_overwrites_only_the_target_line(img_dir, fmt):
    cs = _open(img_dir, fmt)
    cs.set_lines("a.jpg", ["tags", "old caption"])
    cs.write_line("a.jpg", 1, "new caption", "replace")
    assert cs.get_lines("a.jpg") == ["tags", "new caption"]


@pytest.mark.parametrize(
    ("tags", "existing", "new", "expected"),
    [
        (True, "1girl, solo", "solo, smile", "1girl, solo, smile"),  # ", " joined, no repeats
        (False, "A girl.", "She smiles.", "A girl. She smiles."),  # single space
        (True, "", "1girl", "1girl"),  # empty line: same as replace
        (False, "", "Hello.", "Hello."),
    ],
)
def test_append_joins_with_the_right_separator(img_dir, tags, existing, new, expected):
    cs = _open(img_dir)
    cs.set_lines("a.jpg", [existing] if existing else [])
    cs.write_line("a.jpg", 0, new, "append", sep=", " if tags else " ", tags=tags)
    assert cs.get_lines("a.jpg") == [expected]


def test_append_on_a_missing_line_pads_then_behaves_like_replace(img_dir):
    cs = _open(img_dir)
    cs.set_lines("a.jpg", ["tags"])
    cs.write_line("a.jpg", 2, "caption", "append")
    assert cs.get_lines("a.jpg") == ["tags", "", "caption"]


def test_line_has_content_and_unknown_key(img_dir):
    cs = _open(img_dir)
    cs.set_lines("a.jpg", ["x"])
    assert cs.line_has_content("a.jpg", 0)
    assert not cs.line_has_content("a.jpg", 1)
    with pytest.raises(KeyError):
        cs.write_line("nope.jpg", 0, "x")


def test_merge_tags_is_case_insensitive():
    assert merge_tags("Solo, 1girl", "solo, smile") == "Solo, 1girl, smile"


@pytest.mark.parametrize(
    ("write_mode", "overwrite", "expected"),
    [
        ("", False, "skip"),  # a config saved before modes existed
        ("", True, "replace"),  # overwrite = true is replace
        ("append", False, "append"),
        ("append", True, "append"),  # an explicit mode wins over the legacy flag
        ("SKIP", True, "skip"),
    ],
)
def test_effective_write_mode_keeps_old_overwrite_configs(write_mode, overwrite, expected):
    assert effective_write_mode(write_mode, overwrite) == expected


def test_effective_write_mode_rejects_unknown():
    with pytest.raises(ValueError):
        effective_write_mode("merge")


# ------------------------------------------------------------------ config validation


@pytest.mark.parametrize("stage", ["tag", "caption", "edit_caption"])
@pytest.mark.parametrize(
    ("section", "message"),
    [({"target_line": 0}, "target_line"), ({"write_mode": "sideways"}, "write_mode")],
)
def test_validate_rejects_bad_target_line_and_mode(img_dir, stage, section, message):
    config = parse_prep_config({"path": str(img_dir), stage: section})
    with pytest.raises(ValueError, match=message):
        config.validate_for_stage(stage)


# ------------------------------------------------------------------ conversion


def test_sidecar_to_json_gathers_and_removes_sidecars(img_dir):
    (img_dir / "a.txt").write_text("1girl, solo\nA girl.\n")
    (img_dir / "b.txt").write_text("tags only\n")
    (img_dir / "notes.txt").write_text("not a caption")  # no image of that name
    report = convert_captions(img_dir, "sidecar", ".txt", "json", ".txt")
    assert report["converted"] == 2 and report["removed"] == 2
    data = json.loads((img_dir / "captions.json").read_text())
    assert data["a.jpg"] == ["1girl, solo", "A girl."]  # multi-line preserved
    assert data["b.jpg"] == ["tags only"]
    assert not (img_dir / "a.txt").exists() and not (img_dir / "b.txt").exists()
    assert (img_dir / "notes.txt").exists()


def test_json_to_sidecar_writes_sidecars_and_removes_json(img_dir):
    (img_dir / "captions.json").write_text(
        json.dumps({"a.jpg": ["t1, t2", "Caption."], "b.jpg": ["only"], "c.jpg": [""]})
    )
    report = convert_captions(img_dir, "json", ".txt", "sidecar", ".txt")
    assert report["converted"] == 2
    assert (img_dir / "a.txt").read_text() == "t1, t2\nCaption.\n"
    assert (img_dir / "b.txt").read_text() == "only\n"
    assert not (img_dir / "c.txt").exists()  # an empty caption makes no file
    assert not (img_dir / "captions.json").exists()


def test_sidecar_extension_change_renames(img_dir):
    (img_dir / "a.txt").write_text("hello\n")
    convert_captions(img_dir, "sidecar", ".txt", "sidecar", ".caption")
    assert (img_dir / "a.caption").read_text() == "hello\n"
    assert not (img_dir / "a.txt").exists()


def test_padded_empty_lines_survive_conversion(img_dir):
    (img_dir / "a.txt").write_text("\n\nthird\n")
    convert_captions(img_dir, "sidecar", ".txt", "json", ".txt")
    assert json.loads((img_dir / "captions.json").read_text())["a.jpg"] == ["", "", "third"]


def test_conversion_edge_cases_are_noops(tmp_path, img_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert convert_captions(empty, "sidecar", ".txt", "json", ".txt")["converted"] == 0
    assert not (empty / "captions.json").exists()

    # Images with no captions: nothing written, no stray captions.json.
    assert convert_captions(img_dir, "sidecar", ".txt", "json", ".txt")["converted"] == 0
    assert not (img_dir / "captions.json").exists()

    # Same layout: nothing to do.
    (img_dir / "a.txt").write_text("x\n")
    assert convert_captions(img_dir, "sidecar", ".txt", "sidecar", ".txt")["converted"] == 0
    assert (img_dir / "a.txt").exists()


def test_conversion_is_idempotent_and_keeps_unrelated_json_entries(img_dir):
    (img_dir / "captions.json").write_text(json.dumps({"c.jpg": ["kept"]}))
    (img_dir / "a.txt").write_text("from sidecar\n")
    convert_captions(img_dir, "sidecar", ".txt", "json", ".txt")
    convert_captions(img_dir, "sidecar", ".txt", "json", ".txt")  # again: no sidecars left
    data = json.loads((img_dir / "captions.json").read_text())
    assert data["a.jpg"] == ["from sidecar"] and data["c.jpg"] == ["kept"]


# ------------------------------------------------------------------ real runner path


def _stub_tagger(monkeypatch, line="1girl, solo"):
    def fake_chunked(paths, specs, *, on_chunk, on_progress=None, **kwargs):
        on_chunk(paths, {str(p): line for p in paths})

    monkeypatch.setattr("rengu_flow.prep.tagger.run_ensemble_chunked", fake_chunked)


def test_tag_step_converts_sidecar_input_to_json_output(img_dir, tmp_path, monkeypatch):
    """A workflow tag step whose output format is json, fed a folder of sidecars: the folder ends
    with only captions.json, holding the old captions and the new tags."""
    (img_dir / "a.txt").write_text("1girl, standing\nA girl stands.\n")
    _stub_tagger(monkeypatch, "solo, smile")
    config = parse_prep_config(
        {
            "path": str(img_dir),
            "caption_format": "json",  # the step's output layout
            "caption_ext": ".txt",
            "convert_from_format": "sidecar",  # what the folder is in now
            "convert_from_ext": ".txt",
            "tag": {"models": ["pixai-v0.9"], "write_mode": "append"},
        }
    )
    assert run_stage(config, "tag", tmp_path / "job") == 0

    assert not list(img_dir.glob("*.txt"))  # no stale sidecar for the trainer's json-wins rule
    data = json.loads((img_dir / "captions.json").read_text())
    assert data["a.jpg"] == ["1girl, standing, solo, smile", "A girl stands."]
    assert data["b.jpg"] == ["solo, smile"]
    report = json.loads((tmp_path / "job" / "report.json").read_text())
    assert report["caption_conversion"]["converted"] == 1


def test_tag_step_writes_target_line_3_padded(img_dir, tmp_path, monkeypatch):
    _stub_tagger(monkeypatch, "solo")
    config = parse_prep_config(
        {"path": str(img_dir), "tag": {"models": ["pixai-v0.9"], "target_line": 3}}
    )
    assert run_stage(config, "tag", tmp_path / "job") == 0
    assert (img_dir / "a.txt").read_text() == "\n\nsolo\n"
    assert _open(img_dir).get_lines("a.jpg") == ["", "", "solo"]


def test_tag_modes_skip_replace_and_legacy_overwrite(img_dir, tmp_path, monkeypatch):
    _stub_tagger(monkeypatch, "new")
    for name in ("a", "b", "c"):
        (img_dir / f"{name}.txt").write_text("old\n")

    def run(section):
        cfg = parse_prep_config(
            {"path": str(img_dir), "tag": {"models": ["pixai-v0.9"], **section}}
        )
        assert run_stage(cfg, "tag", tmp_path / "job") == 0
        return (img_dir / "a.txt").read_text()

    assert run({}) == "old\n"  # default: skip images that already have the line
    assert run({"write_mode": "skip"}) == "old\n"
    assert run({"overwrite": True}) == "new\n"  # an old config with overwrite = true
    (img_dir / "a.txt").write_text("old\n")
    assert run({"write_mode": "append"}) == "old, new\n"


class _FakeBackend(CaptionBackend):
    def __init__(self, text="Fresh caption."):
        self.text = text

    def load(self):
        pass

    def caption_batch(self, images, prompts):
        return [self.text for _ in prompts]

    def unload(self):
        pass


@pytest.mark.parametrize("fmt", ["sidecar", "json"])
def test_caption_modes_on_arbitrary_line(img_dir, fmt):
    cs = _open(img_dir, fmt)
    cs.set_lines("a.jpg", ["tags", "Old caption."])
    cs.set_lines("b.jpg", ["tags"])
    cs.save()

    def run(text="Fresh caption.", **kwargs):
        return caption_folder(
            img_dir,
            CaptionerConfig(model="joycaption-beta-one", **kwargs),
            fmt=fmt,
            backend_factory=lambda cfg: _FakeBackend(text),
        )

    report = run()  # default skip: a.jpg already has line 2
    assert report["skipped"] == 1 and report["captioned"] == 2
    def lines(key):
        return _open(img_dir, fmt).get_lines(key)

    assert lines("a.jpg") == ["tags", "Old caption."]
    assert lines("b.jpg") == ["tags", "Fresh caption."]

    run("More.", write_mode="append")
    assert lines("a.jpg") == ["tags", "Old caption. More."]
    run("More.", write_mode="append")  # a re-run must not stack the same text again
    assert lines("a.jpg") == ["tags", "Old caption. More."]

    run(write_mode="replace", target_line=4)  # padded: line 3 stays empty
    assert lines("c.jpg") == ["", "Fresh caption. More.", "", "Fresh caption."]
    assert lines("a.jpg")[3] == "Fresh caption."

    run(overwrite=True)  # legacy flag == replace on line 2
    assert lines("a.jpg")[1] == "Fresh caption."


def test_caption_target_line_1_is_allowed(img_dir):
    caption_folder(
        img_dir,
        CaptionerConfig(model="joycaption-beta-one", target_line=1),
        backend_factory=lambda cfg: _FakeBackend(),
    )
    assert _open(img_dir).get_lines("a.jpg") == ["Fresh caption."]


# ---------------------------------------------------------------------- caption tags_line


def _caption_config(tmp_path, **caption):
    config = PrepConfig(path=str(tmp_path))
    for key, value in caption.items():
        setattr(config.caption, key, value)
    return config


@pytest.mark.parametrize("bad", [0, -1, "x", None])
def test_caption_tags_line_must_be_a_positive_whole_number(tmp_path, bad):
    with pytest.raises(ValueError, match="tags_line"):
        _caption_config(tmp_path, tags_line=bad).validate_for_stage("caption")


def test_caption_tags_line_cannot_be_the_line_being_written(tmp_path):
    config = _caption_config(tmp_path, tags_line=2, target_line=2)
    with pytest.raises(ValueError, match="equals target_line"):
        config.validate_for_stage("caption")
    # Grounding off: the setting is inert, nothing to reject.
    _caption_config(tmp_path, tags_line=2, target_line=2, use_tags_as_grounding=False
                    ).validate_for_stage("caption")


@pytest.mark.parametrize(("tags_line", "target_line"), [(1, 2), (3, 2), (1, 1)])
def test_caption_tags_line_valid_combinations(tmp_path, tags_line, target_line):
    # (1, 1) is the legacy default pair and stays valid.
    _caption_config(tmp_path, tags_line=tags_line, target_line=target_line).validate_for_stage("caption")

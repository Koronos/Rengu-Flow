"""Caption layout conversion must never lose captions, and layout mismatches must fail loudly.

Covers the trainer's view of a folder (videos are media too, ``captions.json`` keys may be tar
members or names of files that are gone), the backup taken before anything is removed, the runner's
layout guard, the pre-flight refusal of a sibling reading a folder another step converts, and the
append duplicate guard.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rengu_flow.prep.caption_store import (
    CaptionStore,
    LayoutMismatchError,
    check_layout,
    convert_captions,
)
from rengu_flow.prep.config import parse_prep_config
from rengu_flow.prep.runner import run_stage
from rengu_flow_ui import workflow_graph as wg
from rengu_flow_ui.workflow_graph import WorkflowNode

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


@pytest.fixture(autouse=True)
def _isolate_prep_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(tmp_path / "appdata"))


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    shutil.copy(FIXTURE_JPG, d / "a.png")
    (d / "clip.mp4").write_bytes(b"not really a video")
    return d


def _json(folder):
    return json.loads((folder / "captions.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ json -> sidecar


def test_json_to_sidecar_is_refused_when_entries_cannot_become_sidecars(folder):
    original = {
        "a.png": ["tags"],
        "clip.mp4": ["a clip"],
        "shard/img1.jpg": ["tar member"],  # lives inside a tar
        "gone.png": ["file no longer here"],
    }
    (folder / "captions.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot become sidecar files"):
        convert_captions(folder, "json", ".txt", "sidecar", ".txt")

    # Refused BEFORE anything was written or removed.
    assert _json(folder) == original
    assert not list(folder.glob("*.txt"))


def test_json_to_sidecar_covers_videos_and_backs_captions_json_up(folder, tmp_path):
    (folder / "captions.json").write_text(
        json.dumps({"a.png": ["t", "caption"], "clip.mp4": ["a clip"]}), encoding="utf-8"
    )

    report = convert_captions(folder, "json", ".txt", "sidecar", ".txt")

    assert (folder / "a.txt").read_text() == "t\ncaption\n"
    assert (folder / "clip.txt").read_text() == "a clip\n"  # the video is media too
    assert not (folder / "captions.json").exists()
    backup = Path(report["backup"])
    assert json.loads((backup / "captions.json").read_text())["clip.mp4"] == ["a clip"]
    assert (backup / "manifest.json").is_file()


# ------------------------------------------------------------------------------ sidecar -> json


def test_sidecar_to_json_merges_into_the_existing_file_and_includes_videos(folder):
    (folder / "captions.json").write_text(
        json.dumps({"other.mp4": ["keep me"], "shard/img1.jpg": ["tar member"]}), encoding="utf-8"
    )
    (folder / "a.txt").write_text("tags\n")
    (folder / "clip.txt").write_text("clip caption\n")

    report = convert_captions(folder, "sidecar", ".txt", "json", ".txt")

    data = _json(folder)
    assert data["other.mp4"] == ["keep me"]  # unrelated keys survive
    assert data["shard/img1.jpg"] == ["tar member"]
    assert data["a.png"] == ["tags"] and data["clip.mp4"] == ["clip caption"]
    # No stale sidecar is left for training to ignore (json wins there), and they are recoverable.
    assert not (folder / "a.txt").exists() and not (folder / "clip.txt").exists()
    backup = Path(report["backup"])
    assert (backup / "a.txt").read_text() == "tags\n"
    assert (backup / "clip.txt").read_text() == "clip caption\n"


def test_invalid_existing_json_is_refused_and_nothing_is_removed(folder):
    (folder / "captions.json").write_text("{ not json", encoding="utf-8")
    (folder / "a.txt").write_text("tags\n")

    with pytest.raises(ValueError, match="not valid JSON"):
        convert_captions(folder, "sidecar", ".txt", "json", ".txt")

    assert (folder / "a.txt").read_text() == "tags\n"


# ------------------------------------------------------------------------------ layout guard


def test_check_layout_names_the_layout_the_folder_is_really_in(folder):
    (folder / "captions.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LayoutMismatchError, match="captions.json"):
        check_layout(folder, "sidecar", ".txt")
    check_layout(folder, "json", ".txt")  # consistent

    (folder / "captions.json").unlink()
    (folder / "a.txt").write_text("x\n")
    with pytest.raises(LayoutMismatchError, match="sidecar layout"):
        check_layout(folder, "json", ".txt")
    check_layout(folder, "sidecar", ".txt")


def test_a_stage_in_the_wrong_layout_fails_and_writes_nothing(folder, tmp_path, monkeypatch):
    """A sibling step that kept the pre-conversion handle: sidecar mode in a json folder."""
    (folder / "captions.json").write_text(json.dumps({"a.png": ["tags"]}), encoding="utf-8")

    def fake_chunked(paths, specs, *, on_chunk, **kwargs):
        on_chunk(paths, {str(p): "new" for p in paths})

    monkeypatch.setattr("rengu_flow.prep.tagger.run_ensemble_chunked", fake_chunked)
    config = parse_prep_config({"path": str(folder), "tag": {"models": ["pixai-v0.9"]}})

    assert run_stage(config, "tag", tmp_path / "job") == 1
    report = json.loads((tmp_path / "job" / "report.json").read_text())
    assert "captions.json" in report["error"]
    assert not list(folder.glob("*.txt"))  # nothing a trainer would ignore was written


# ------------------------------------------------------------------------------ pre-flight


def _n(node_id, node_type, source=None, **config):
    return WorkflowNode(id=node_id, type=node_type, source=source, config=config)


def _tag(node_id, source, **config):
    return _n(node_id, "prep.tag", source, models=["pixai-v0.9"], **config)


def test_a_sibling_of_a_converting_step_is_refused():
    graph = wg.WorkflowGraph(
        nodes=[
            _n("f", "folder", path="."),
            _tag("a", "f", output_format="json"),
            _n("b", "prep.caption", "f", model="joycaption-beta-one"),  # reads f, not a
        ]
    )
    errors = wg.validate(graph)
    assert any("node b" in e and "converts" in e for e in errors), errors


def test_downstream_and_earlier_steps_of_a_converter_are_fine():
    ok = wg.WorkflowGraph(
        nodes=[
            _n("f", "folder", path="."),
            _tag("early", "f"),  # runs before the conversion
            _tag("a", "f", output_format="json"),
            _n("b", "prep.caption", "a", model="joycaption-beta-one"),  # downstream: json handle
        ]
    )
    assert not any("converts" in e for e in wg.validate(ok))


def test_a_step_that_keeps_the_layout_converts_nothing():
    graph = wg.WorkflowGraph(
        nodes=[
            _n("f", "folder", path=".", caption_format="json"),
            _tag("a", "f", output_format="json"),
            _n("b", "prep.caption", "f", model="joycaption-beta-one"),
        ]
    )
    assert not any("converts" in e for e in wg.validate(graph))


# ------------------------------------------------------------------------------ edit after a tool


def test_edit_caption_after_a_folder_tool_does_not_fail_for_a_control_folder_it_cannot_know():
    graph = wg.WorkflowGraph(
        nodes=[
            _n("t", "tool", tool_id="extract"),
            _n("e", "prep.edit_caption", "t", model="qwen3-vl-4b-instruct"),
        ]
    )
    tool_io = {"extract": {"input": "none", "output": "folder", "output_declared": True}}
    assert not any("control images folder" in e for e in wg.validate(graph, tool_io=tool_io))
    # A known folder source is still judged as before.
    plain = wg.WorkflowGraph(
        nodes=[
            _n("f", "folder", path="."),
            _n("e", "prep.edit_caption", "f", model="qwen3-vl-4b-instruct"),
        ]
    )
    assert any("control images folder" in e for e in wg.validate(plain))


# ------------------------------------------------------------------------------ append + dump


def test_append_does_not_stack_identical_text(folder):
    cs = CaptionStore.open(folder, positional=True)
    cs.write_line("a.png", 1, "Same caption.", "append")
    cs.write_line("a.png", 1, "Same caption.", "append")
    assert cs.get_lines("a.png") == ["", "Same caption."]
    cs.write_line("a.png", 1, "Another.", "append")
    assert cs.get_lines("a.png") == ["", "Same caption. Another."]


def test_dump_dataset_ignores_blank_json_variants_like_the_loader(folder):
    from rengu_flow.data.dump_dataset import _caption_for_image

    captions = {"a.png": ["", "", "third"], "b.png": ["", ""]}
    assert _caption_for_image(folder / "a.png", {}, captions) == ["third"]
    assert _caption_for_image(folder / "b.png", {}, captions) == [""]


# ------------------------------------------------------------------------------ review round 2


def test_json_mode_ignores_leftover_sidecars_when_captions_json_exists(folder):
    """The trainer ignores sidecars once captions.json exists: nothing to protect, nothing to refuse."""
    (folder / "captions.json").write_text(json.dumps({"a.png": ["x"]}), encoding="utf-8")
    (folder / "a.txt").write_text("leftover\n")
    check_layout(folder, "json", ".txt")


def test_json_mode_only_counts_sidecars_of_media_files(folder):
    (folder / "notes.md").write_text("notes")
    (folder / "notes.txt").write_text("just some notes\n")
    check_layout(folder, "json", ".txt")  # no captions.json, but notes.txt is nobody's caption
    (folder / "a.txt").write_text("caption\n")
    with pytest.raises(LayoutMismatchError):
        check_layout(folder, "json", ".txt")


def test_json_to_sidecar_backs_up_the_sidecars_it_overwrites(folder):
    (folder / "a.txt").write_text("hand written\n")
    (folder / "captions.json").write_text(json.dumps({"a.png": ["from json"]}), encoding="utf-8")

    report = convert_captions(folder, "json", ".txt", "sidecar", ".txt")

    assert (folder / "a.txt").read_text() == "from json\n"
    backup = Path(report["backup"])
    assert (backup / "a.txt").read_text() == "hand written\n"  # the overwritten file is recoverable
    assert (backup / "captions.json").is_file()


def test_sidecar_rename_backs_up_sources_and_overwritten_targets(folder):
    (folder / "a.txt").write_text("source\n")
    (folder / "a.caption").write_text("existing target\n")

    report = convert_captions(folder, "sidecar", ".txt", "sidecar", ".caption")

    assert (folder / "a.caption").read_text() == "source\n"
    backup = Path(report["backup"])
    assert (backup / "a.caption").read_text() == "existing target\n"
    assert (backup / "a.txt").read_text() == "source\n"


def test_sidecar_to_json_backs_up_the_json_it_merges_into(folder):
    (folder / "captions.json").write_text(json.dumps({"x.mp4": ["keep"]}), encoding="utf-8")
    (folder / "a.txt").write_text("t\n")
    report = convert_captions(folder, "sidecar", ".txt", "json", ".txt")
    assert json.loads((Path(report["backup"]) / "captions.json").read_text()) == {"x.mp4": ["keep"]}


def test_two_folder_steps_naming_the_same_directory_are_one_folder():
    graph = wg.WorkflowGraph(
        nodes=[
            _n("f1", "folder", path="D:/data/set"),
            _n("f2", "folder", path="D:/data/set/"),
            _tag("a", "f1", output_format="json"),
            _n("b", "prep.caption", "f2", model="joycaption-beta-one"),
        ]
    )
    assert any("node b" in e and "converts" in e for e in wg.validate(graph))

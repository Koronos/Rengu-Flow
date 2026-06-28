"""Quality index (SQLite) logic, GPU-free via an injected score function."""

import os
from pathlib import Path

import pytest

from rengu_flow.prep import quality_index as qi


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("RENGU_FLOW_UI_DATA", str(tmp_path / "data"))
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(10):
        (d / f"img{i:02d}.png").write_bytes(b"x")
    return d


def make_score_fn(quals):
    """Deterministic, call-recording stand-in for the GPU scorers.

    ``quals`` maps model -> {basename: quality}. Records which (model, names) it
    was asked to score so tests can assert nothing is rescored.
    """
    calls = []

    def fn(model, paths, should_stop):
        calls.append((model, [Path(p).name for p in paths]))
        for p in paths:
            q = quals[model][Path(p).name]
            yield p, q, q  # (path, raw, quality)

    fn.calls = calls
    return fn


def _linear(n=10):
    """img00 (worst, q=0) .. img09 (best, q=9)."""
    return {f"img{i:02d}.png": float(i) for i in range(n)}


def test_build_then_incremental_skips_scored(dataset):
    fn = make_score_fn({"m": _linear()})
    rep = qi.build_index(dataset, ["m"], score_fn=fn)
    assert rep == {"images": 10, "models": {"m": 10}}

    fn2 = make_score_fn({"m": _linear()})
    rep2 = qi.build_index(dataset, ["m"], score_fn=fn2)
    assert rep2["models"]["m"] == 0
    assert fn2.calls == []  # already-indexed pairs are never rescored


def test_worst_returns_lowest_quality_ascending(dataset):
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    worst = qi.worst(dataset, "m", 3)
    assert [Path(w["path"]).name for w in worst] == ["img00.png", "img01.png", "img02.png"]
    assert [w["quality"] for w in worst] == [0.0, 1.0, 2.0]


def test_cull_preview_unions_per_model(dataset):
    quals = {
        "a": _linear(),  # img00..01 are a's worst
        "b": {f"img{i:02d}.png": float(9 - i) for i in range(10)},  # img09..08 are b's worst
    }
    qi.build_index(dataset, ["a", "b"], score_fn=make_score_fn(quals))
    cp = qi.cull_preview(dataset, {"a": 20, "b": 20})
    assert cp["per_model"] == {"a": 2, "b": 2}
    assert cp["union"] == 4  # disjoint sets -> 2 + 2
    assert sorted(Path(p).name for p in cp["paths"]) == [
        "img00.png", "img01.png", "img08.png", "img09.png"
    ]


def test_changed_mtime_triggers_rescore(dataset):
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    target = dataset / "img05.png"
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 1000))

    fn = make_score_fn({"m": _linear()})
    rep = qi.build_index(dataset, ["m"], score_fn=fn)
    assert rep["models"]["m"] == 1
    assert fn.calls == [("m", ["img05.png"])]


def test_cull_does_not_erode_on_rerun(dataset):
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    cp = qi.cull_preview(dataset, {"m": 30})
    assert cp["union"] == 3  # img00, img01, img02

    # Simulate culling: remove the flagged files from the folder.
    for p in cp["paths"]:
        Path(p).unlink()

    # Rebuilding keeps the removed images in the reference (present=0), so the
    # percentile cutoff is unchanged and a second cull at the same percent drops
    # nothing new.
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    cp2 = qi.cull_preview(dataset, {"m": 30})
    assert cp2["present"] == 7
    assert cp2["union"] == 0


def test_apply_cull_moves_union_and_is_idempotent(dataset):
    (dataset / "img00.png").with_suffix(".txt").write_text("caption")  # sidecar moves too
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))

    res = qi.apply_cull(dataset, {"m": 30})
    assert res["moved"] == 3
    low = dataset / "low_quality"
    assert sorted(p.name for p in low.iterdir()) == [
        "img00.png", "img00.txt", "img01.png", "img02.png"
    ]

    # Re-applying the same cull drops nothing: the moved images stay in the
    # reference (present=0) so the cutoff is unchanged.
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    assert qi.apply_cull(dataset, {"m": 30})["moved"] == 0


def test_new_images_score_only_the_additions(dataset):
    qi.build_index(dataset, ["m"], score_fn=make_score_fn({"m": _linear()}))
    (dataset / "img10.png").write_bytes(b"x")
    quals = {"m": {**_linear(), "img10.png": 4.5}}

    fn = make_score_fn(quals)
    rep = qi.build_index(dataset, ["m"], score_fn=fn)
    assert rep["models"]["m"] == 1
    assert fn.calls == [("m", ["img10.png"])]

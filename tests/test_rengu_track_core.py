"""Unit tests for the rengu_track core: manifest, event timeline, and sink fan-out.

These touch only the filesystem (no torch, no UI DB), so the module opts out of the autouse UI
sqlite fixture.
"""

import json

import pytest

from rengu_track import (
    EVENT_CONFIG_RELOADED,
    EVENT_RUN_STARTED,
    NullSink,
    append_event,
    build_sink,
    config_diff,
    read_events,
)
from rengu_track.run import (
    RunManifest,
    flatten_hparams,
    read_manifest,
    write_manifest,
)
from rengu_track.sink import MetricsSink

pytestmark = pytest.mark.no_ui_db


# --- manifest ---------------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    manifest = RunManifest(
        run_id="20260101_00-00-00",
        name="run-a",
        config={"optimizer": {"lr": 1e-4}, "model": {"type": "sdxl"}},
    )
    write_manifest(tmp_path, manifest)

    assert (tmp_path / "run.json").is_file()
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded.run_id == "20260101_00-00-00"
    assert loaded.name == "run-a"
    assert loaded.config["optimizer"]["lr"] == 1e-4
    assert loaded.updated_at  # stamped on write


def test_read_manifest_missing_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None


def test_write_manifest_stringifies_non_json_config(tmp_path):
    """The live training config holds torch.dtype values post-defaults; the manifest
    must stringify them instead of failing the run at sink construction."""
    torch = pytest.importorskip("torch")
    manifest = RunManifest(
        run_id="r1",
        config={"model": {"type": "sdxl", "dtype": torch.bfloat16}},
    )
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded.config["model"]["dtype"] == "torch.bfloat16"


def test_from_dict_drops_unknown_keys(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps({"run_id": "x", "name": "n", "some_future_field": 1}),
        encoding="utf-8",
    )
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded.run_id == "x"


def test_flatten_hparams_scalars_lists_and_nested():
    flat = flatten_hparams(
        {
            "optimizer": {"type": "adamw", "lr": 1e-4},
            "resolutions": [512, 1024],
            "enabled": True,
            "nested": {"a": {"b": 3}},
        }
    )
    assert flat["optimizer.type"] == "adamw"
    assert flat["optimizer.lr"] == 1e-4
    assert flat["resolutions"] == "512, 1024"  # scalar list -> joined string column
    assert flat["enabled"] is True
    assert flat["nested.a.b"] == 3


def test_flatten_hparams_list_of_dicts_becomes_indexed_rows():
    # A list of dicts must NOT collapse into one repr blob; each leaf gets its own column.
    flat = flatten_hparams(
        {"schedule": {"stage": [{"res": [512, 1024], "frac": 0.4}, {"res": [1024], "frac": 0.2}]}}
    )
    assert flat["schedule.stage[0].res"] == "512, 1024"
    assert flat["schedule.stage[0].frac"] == 0.4
    assert flat["schedule.stage[1].res"] == "1024"
    assert flat["schedule.stage[1].frac"] == 0.2
    assert not any("{" in str(v) for v in flat.values())  # no inline-dict blobs survive


# --- events -----------------------------------------------------------------------------------


def test_append_and_read_events(tmp_path):
    append_event(tmp_path, EVENT_RUN_STARTED, step=0, payload={"resume": False})
    append_event(tmp_path, EVENT_CONFIG_RELOADED, step=42, payload={"changed": {}}, source="trainer")

    events = read_events(tmp_path)
    assert [e["type"] for e in events] == [EVENT_RUN_STARTED, EVENT_CONFIG_RELOADED]
    assert events[0]["step"] == 0
    assert events[1]["step"] == 42
    assert all(e["ts"] for e in events)


def test_read_events_skips_malformed_lines(tmp_path):
    path = tmp_path / "run_events.jsonl"
    path.write_text('{"type":"run_started","ts":"t","step":0}\nnot json\n\n', encoding="utf-8")
    events = read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["type"] == "run_started"


def test_config_diff():
    old = {"optimizer": {"lr": 1e-4}, "preview": {"every": 100}, "dropme": 1}
    new = {"optimizer": {"lr": 2e-4}, "preview": {"every": 100}, "added": 5}
    diff = config_diff(old, new)
    assert diff["changed"] == {"optimizer.lr": [1e-4, 2e-4]}
    assert diff["added"] == {"added": 5}
    assert diff["removed"] == {"dropme": 1}


# --- sink fan-out -----------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, fail_on=None):
        self.calls = []
        self._fail_on = fail_on

    def _record(self, name, *args, **kwargs):
        if name == self._fail_on:
            raise RuntimeError(f"boom in {name}")
        self.calls.append((name, args, kwargs))

    def scalar(self, *a, **k):
        self._record("scalar", *a, **k)

    def histogram(self, *a, **k):
        self._record("histogram", *a, **k)

    def image(self, *a, **k):
        self._record("image", *a, **k)

    def set_metadata(self, *a, **k):
        self._record("set_metadata", *a, **k)

    def summary(self, *a, **k):
        self._record("summary", *a, **k)

    def close(self, *a, **k):
        self._record("close", *a, **k)


def test_sink_fans_out_to_all_backends(tmp_path):
    a, b = FakeBackend(), FakeBackend()
    sink = MetricsSink([a, b], tmp_path)

    sink.scalar("train/loss", 0.5, 1)
    sink.set_hparams({"lr": 1e-4})
    sink.summary({"best_loss": 0.1})
    sink.close(status="finished")

    for backend in (a, b):
        names = [c[0] for c in backend.calls]
        assert names == ["scalar", "set_metadata", "summary", "close"]
    # scalar value is coerced to float
    assert a.calls[0] == ("scalar", ("train/loss", 0.5, 1), {})
    assert a.calls[1][2] == {"config": {"lr": 1e-4}}


def test_sink_isolates_backend_failure(tmp_path):
    bad = FakeBackend(fail_on="scalar")
    good = FakeBackend()
    sink = MetricsSink([bad, good], tmp_path)

    # Must not raise even though `bad.scalar` throws.
    sink.scalar("train/loss", 0.5, 1)
    sink.scalar("train/loss", 0.4, 2)

    assert [c[0] for c in good.calls] == ["scalar", "scalar"]


def test_sink_event_appends_timeline(tmp_path):
    sink = MetricsSink([], tmp_path)
    sink.event(EVENT_RUN_STARTED, step=0, payload={"resume": False})
    events = read_events(tmp_path)
    assert len(events) == 1
    assert events[0]["type"] == EVENT_RUN_STARTED


# --- build_sink -------------------------------------------------------------------------------


def test_build_sink_disabled_returns_nullsink(tmp_path):
    sink = build_sink({"tracking": {"enabled": False}}, tmp_path)
    assert isinstance(sink, NullSink)
    # No artifacts written.
    sink.scalar("train/loss", 1.0, 1)
    sink.event(EVENT_RUN_STARTED)
    assert not (tmp_path / "run.json").exists()
    assert not (tmp_path / "run_events.jsonl").exists()


def test_build_sink_manifest_only_writes_run_json(tmp_path):
    # Select only the manifest backend so the test never imports torch/tensorboard.
    config = {"tracking": {"backends": ["manifest"]}, "optimizer": {"lr": 1e-4}}
    sink = build_sink(config, tmp_path)
    assert isinstance(sink, MetricsSink)

    assert (tmp_path / "run.json").is_file()
    manifest = read_manifest(tmp_path)
    assert manifest.run_id == tmp_path.name
    assert manifest.hparams_flat["optimizer.lr"] == 1e-4

    sink.summary({"best_loss": 0.2, "system/peak_vram_gb": 7.5})
    sink.close(status="finished")
    manifest = read_manifest(tmp_path)
    assert manifest.summary["best_loss"] == 0.2
    assert manifest.system_summary["peak_vram_gb"] == 7.5
    assert manifest.status == "finished"


def test_manifest_backend_records_scalar_index(tmp_path):
    # The manifest backend records the last value per scalar tag (cheap index for lists/compare),
    # flushed to run.json on close — without storing the full series.
    sink = build_sink({"tracking": {"backends": ["manifest"]}}, tmp_path)
    sink.scalar("train/loss", 0.5, 1)
    sink.scalar("train/loss", 0.3, 2)
    sink.scalar("val/loss", 0.7, 2)
    sink.close(status="finished")

    manifest = read_manifest(tmp_path)
    assert manifest.scalar_tags == ["train/loss", "val/loss"]
    assert manifest.last_scalars["train/loss"] == 0.3
    assert manifest.last_scalars["val/loss"] == 0.7

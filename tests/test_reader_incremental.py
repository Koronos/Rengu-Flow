"""The scalar reader keeps a persistent EventAccumulator and reloads incrementally.

Recreating the accumulator on every call re-decoded the whole event stream (~4.5s on a real run,
dominated by embedded preview-image bytes) on every poll of a live run. Holding it alive and
calling Reload() — which continues from the last file offset — drops repeat reads to ~1ms, matching
how TensorBoard stays fast on the same files.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.no_ui_db

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from rengu_track import reader
from rengu_track.backends.tensorboard import TensorBoardBackend


def test_reader_picks_up_appended_scalars(tmp_path: Path) -> None:
    """A second read after more scalars are written must reflect them (file mtime moved → re-read),
    exercising the persistent accumulator's incremental Reload."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reader.invalidate_scalars_cache()
    backend = TensorBoardBackend(run_dir)
    backend.scalar("train/loss", 0.5, step=1)
    backend._writer.flush()

    first = reader.read_scalars(run_dir)["train/loss"]
    assert [p["step"] for p in first] == [1]

    backend.scalar("train/loss", 0.4, step=2)
    backend._writer.flush()

    second = reader.read_scalars(run_dir)["train/loss"]
    assert [p["step"] for p in second] == [1, 2]
    assert second[-1]["value"] == pytest.approx(0.4)
    backend.close()
    reader.invalidate_scalars_cache()

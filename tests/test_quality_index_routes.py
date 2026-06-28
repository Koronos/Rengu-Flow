"""Quality-index web routes over a GPU-free, pre-populated SQLite index."""

from pathlib import Path

import pytest

P = "/api/v1/prep/quality-index"


def _score_fn(model, paths, should_stop):
    # quality from the filename digits: img00 -> 0 (worst) .. img09 -> 9 (best)
    for p in paths:
        q = float(Path(p).name[3:5])
        yield p, q, q


@pytest.fixture
def indexed(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(10):
        (d / f"img{i:02d}.png").write_bytes(b"x")
    return d


def test_quality_index_routes(ui_client, indexed):
    from rengu_flow.prep import quality_index as qi

    # ui_client's ui_data_tmp points RENGU_FLOW_UI_DATA at tmp, so the index db
    # the build writes is the same one the routes read.
    qi.build_index(indexed, ["m"], score_fn=_score_fn)

    stats = ui_client.get(f"{P}/stats", params={"path": str(indexed), "model": "m"})
    assert stats.status_code == 200, stats.text
    assert stats.json()["reference"] == 10
    assert stats.json()["present"] == 10

    worst = ui_client.get(
        f"{P}/worst", params={"path": str(indexed), "model": "m", "limit": 3}
    ).json()
    assert [i["name"] for i in worst["items"]] == ["img00.png", "img01.png", "img02.png"]
    assert worst["items"][0]["token"]  # signed image token for the thumbnail

    preview = ui_client.post(
        f"{P}/cull-preview", json={"path": str(indexed), "per_model": {"m": 30}}
    ).json()
    assert preview["union"] == 3
    assert preview["present"] == 10
    assert preview["cutoffs"]["m"] == 3.0  # thumbnails with quality < 3 are marked for removal

    applied = ui_client.post(
        f"{P}/apply", json={"path": str(indexed), "per_model": {"m": 30}}
    ).json()
    assert applied["moved"] == 3
    assert sorted(p.name for p in (indexed / "low_quality").iterdir()) == [
        "img00.png", "img01.png", "img02.png"
    ]

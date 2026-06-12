"""tag_ops: filters, scopes, add/remove/rename/prune/quarantine, determinism."""

import pytest

from rengu_flow.prep.tag_ops import (
    TagEditOp,
    TagFilter,
    apply_ops,
    diff_captions,
    is_tag_line,
    line_indices_for_scope,
    replace_underscores,
    select_images,
    tag_frequencies,
)

pytestmark = pytest.mark.no_ui_db


CAPS = {
    "a.jpg": ["1girl, long hair, smile", "A girl with long hair smiles at the camera."],
    "b.jpg": ["1girl, short hair", "1girl, short hair, indoors"],  # tag variants
    "c.jpg": ["2girls, long hair"],
    "d.jpg": [],
}


def op(d):
    return TagEditOp.from_dict(d)


# -- heuristics ------------------------------------------------------------------


def test_is_tag_line():
    assert is_tag_line("1girl, long hair, smile")
    assert is_tag_line("1girl")
    assert is_tag_line("looking at viewer, blue eyes, school uniform")
    assert not is_tag_line("A girl with long hair smiles at the camera.")
    assert not is_tag_line("She smiles, holding an umbrella, while rain falls.")
    assert not is_tag_line("")


def test_replace_underscores_kaomoji_safe():
    assert replace_underscores("long_hair") == "long hair"
    assert replace_underscores("^_^") == "^_^"
    assert replace_underscores("0_0") == "0_0"


def test_line_indices_scopes():
    lines = CAPS["a.jpg"]
    assert line_indices_for_scope(lines, "line1") == [0]
    assert line_indices_for_scope(lines, "tag_lines") == [0]
    assert line_indices_for_scope(lines, "all_lines") == [0, 1]
    assert line_indices_for_scope(lines, "line_n", 1) == [1]
    assert line_indices_for_scope(lines, "line_n", 5) == []
    assert line_indices_for_scope(CAPS["b.jpg"], "tag_lines") == [0, 1]
    with pytest.raises(ValueError):
        line_indices_for_scope(lines, "bogus")


# -- filters ---------------------------------------------------------------------


def test_filter_all_any_none_case_insensitive():
    f = TagFilter.from_dict({"all": ["1GIRL"], "none": ["short hair"]})
    assert select_images(CAPS, f) == ["a.jpg"]
    f = TagFilter.from_dict({"any": ["short hair", "2girls"]})
    assert select_images(CAPS, f) == ["b.jpg", "c.jpg"]
    f = TagFilter.from_dict({"all": ["1girl", "indoors"]})
    assert select_images(CAPS, f) == ["b.jpg"]  # variant line counts (tag_lines scope)


def test_frequencies_line1_vs_all_tag_lines():
    line1 = tag_frequencies(CAPS, scope="line1")
    assert line1["1girl"] == 2 and line1["long hair"] == 2
    assert "indoors" not in line1
    tag_lines = tag_frequencies(CAPS, scope="tag_lines")
    assert tag_lines["indoors"] == 1  # picked up from b.jpg's variant line


# -- ops ---------------------------------------------------------------------------


def test_add_with_filter_and_position():
    res = apply_ops(
        CAPS,
        [op({"op": "add", "tags": ["masterpiece"], "filter": {"all": ["1girl"]},
             "scope": "line1", "position": "start"})],
    )
    assert res.captions["a.jpg"][0] == "masterpiece, 1girl, long hair, smile"
    assert res.captions["b.jpg"][0] == "masterpiece, 1girl, short hair"
    assert res.captions["c.jpg"][0] == "2girls, long hair"  # filtered out
    assert res.changed_keys == ["a.jpg", "b.jpg"]
    # NL caption line untouched
    assert res.captions["a.jpg"][1] == CAPS["a.jpg"][1]


def test_add_is_idempotent_per_line():
    res = apply_ops(CAPS, [op({"op": "add", "tags": ["1girl"], "scope": "line1"})])
    assert res.captions["a.jpg"][0] == CAPS["a.jpg"][0]  # already present, no dup
    assert res.captions["c.jpg"][0] == "2girls, long hair, 1girl"


def test_add_to_empty_or_prose_only_inserts_new_tag_line():
    caps = {"x.jpg": [], "y.jpg": ["A photo of a cat sleeping on a couch."]}
    res = apply_ops(caps, [op({"op": "add", "tags": ["cat", "solo"]})])
    assert res.captions["x.jpg"] == ["cat, solo"]
    assert res.captions["y.jpg"] == ["cat, solo", "A photo of a cat sleeping on a couch."]


def test_remove_propagates_to_variant_lines_with_tag_lines_scope():
    res = apply_ops(CAPS, [op({"op": "remove", "tags": ["short hair"]})])
    assert res.captions["b.jpg"] == ["1girl", "1girl, indoors"]
    # line1 scope only touches the canonical line
    res = apply_ops(CAPS, [op({"op": "remove", "tags": ["short hair"], "scope": "line1"})])
    assert res.captions["b.jpg"] == ["1girl", "1girl, short hair, indoors"]


def test_rename_dedupes_when_target_present():
    caps = {"x.jpg": ["longhair, long hair, 1girl"]}
    res = apply_ops(caps, [op({"op": "rename", "tags": ["longhair"], "rename_to": "long hair"})])
    assert res.captions["x.jpg"] == ["long hair, 1girl"]


def test_prune_below_min_count():
    res = apply_ops(CAPS, [op({"op": "prune", "min_count": 2, "scope": "line1"})])
    # smile (1), short hair (1), 2girls (1) pruned; 1girl (2) and long hair (2) stay
    assert res.captions["a.jpg"][0] == "1girl, long hair"
    assert res.captions["b.jpg"][0] == "1girl"
    assert res.captions["c.jpg"][0] == "long hair"


def test_quarantine_selects_and_drops():
    res = apply_ops(CAPS, [op({"op": "quarantine", "filter": {"any": ["2girls"]}})])
    assert res.quarantined == ["c.jpg"]
    assert "c.jpg" not in res.captions
    assert "a.jpg" in res.captions


def test_apply_is_pure_and_deterministic():
    before = {k: list(v) for k, v in CAPS.items()}
    ops = [
        op({"op": "add", "tags": ["new"], "scope": "line1"}),
        op({"op": "remove", "tags": ["smile"]}),
    ]
    r1 = apply_ops(CAPS, ops)
    r2 = apply_ops(CAPS, ops)
    assert CAPS == before  # input untouched
    assert r1.captions == r2.captions and r1.changed_keys == r2.changed_keys


def test_diff_captions():
    res = apply_ops(CAPS, [op({"op": "remove", "tags": ["smile"], "scope": "line1"})])
    diff = diff_captions(CAPS, res.captions)
    assert len(diff) == 1
    assert diff[0]["key"] == "a.jpg"
    assert diff[0]["before"][0].endswith("smile")
    assert diff[0]["after"][0] == "1girl, long hair"


# -- validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "explode"},
        {"op": "add", "tags": []},
        {"op": "rename", "tags": ["a"]},
        {"op": "prune"},
        {"op": "quarantine"},
        {"op": "add", "tags": ["a"], "scope": "bogus"},
    ],
)
def test_from_dict_validation(bad):
    with pytest.raises(ValueError):
        TagEditOp.from_dict(bad)


def test_op_roundtrip_dict():
    original = {"op": "add", "tags": ["x"], "filter": {"all": ["y"]}, "scope": "line1",
                "position": "start"}
    assert TagEditOp.from_dict(TagEditOp.from_dict(original).to_dict()) == TagEditOp.from_dict(
        original
    )


def test_quarantine_with_explicit_keys():
    res = apply_ops(CAPS, [op({"op": "quarantine", "keys": ["a.jpg", "missing.jpg"]})])
    assert res.quarantined == ["a.jpg"]
    assert "a.jpg" not in res.captions


def test_explicit_keys_intersect_with_filter():
    res = apply_ops(
        CAPS,
        [op({"op": "remove", "tags": ["long hair"], "keys": ["a.jpg", "c.jpg"],
             "filter": {"all": ["1girl"]}, "scope": "line1"})],
    )
    # c.jpg matches keys but not the 1girl filter -> untouched.
    assert res.captions["a.jpg"][0] == "1girl, smile"
    assert res.captions["c.jpg"][0] == "2girls, long hair"

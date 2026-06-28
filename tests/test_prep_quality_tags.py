"""The aesthetic quality-tag mapping must cover every deepghs aesthetic label."""

from rengu_flow.prep.quality import AESTHETIC_LABELS
from rengu_flow.prep.runner import AESTHETIC_QUALITY_TAGS


def test_quality_tag_mapping_covers_all_labels():
    assert set(AESTHETIC_QUALITY_TAGS) == set(AESTHETIC_LABELS)
    # higher tiers keep their booru tag; the cull labels become "<x> quality"
    assert AESTHETIC_QUALITY_TAGS["masterpiece"] == "masterpiece"
    assert AESTHETIC_QUALITY_TAGS["worst"] == "worst quality"

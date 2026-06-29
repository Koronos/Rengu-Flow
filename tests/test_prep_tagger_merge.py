"""Tests for tagger.py: merge logic, run_ensemble seam, registry sanity.

No onnxruntime or GPU required — all heavy inference is replaced by fakes
injected through the ``infer_factory`` parameter.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rengu_flow.prep.tagger import (
    KNOWN_TAGGERS,
    TaggerModelSpec,
    merge_model_results,
    run_ensemble,
)

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_cc0" / "images" / "gb82_01.jpg"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_tmp_images(tmp_path: Path) -> list[Path]:
    """Three copies of the fixture JPEG in a temp directory."""
    imgs = []
    for i in range(3):
        dst = tmp_path / f"img_{i}.jpg"
        shutil.copy(FIXTURE_JPG, dst)
        imgs.append(dst)
    return imgs


def _make_spec(id_: str = "fake-model") -> TaggerModelSpec:
    return TaggerModelSpec(
        id=id_,
        repo_id=f"fake/{id_}",
        filename="model.onnx",
        tags_filename="tags.csv",
        general_threshold=0.35,
        character_threshold=0.85,
        rating_threshold=0.50,
    )


# ---------------------------------------------------------------------------
# merge_model_results: pure merge logic
# ---------------------------------------------------------------------------

class TestMergeModelResults:
    """All tests operate on pre-built {key: {tag: prob}} dicts — no I/O, no GPU."""

    def test_max_prob_wins_across_models(self):
        """When two models disagree on a tag probability, the higher one wins."""
        model_a = {"img.jpg": {"cat": 0.9, "dog": 0.4}}
        model_b = {"img.jpg": {"cat": 0.5, "dog": 0.8, "bird": 0.6}}
        result = merge_model_results([model_a, model_b])
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        # dog: max(0.4, 0.8)=0.8; cat: max(0.9, 0.5)=0.9; bird=0.6
        # sorted desc by prob: cat(0.9), dog(0.8), bird(0.6)
        assert tags == ["cat", "dog", "bird"]

    def test_sorted_by_probability_descending(self):
        model = {"img.jpg": {"a": 0.3, "b": 0.9, "c": 0.6, "d": 0.1}}
        result = merge_model_results([model])
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["b", "c", "a", "d"]

    def test_exclude_tags_case_insensitive(self):
        model = {"img.jpg": {"cat": 0.9, "Dog": 0.8, "bird": 0.6}}
        result = merge_model_results([model], exclude_tags=["dog", "BIRD"])
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["cat"]
        assert "Dog" not in tags
        assert "bird" not in tags

    def test_prepend_tags_lead_and_dedup(self):
        """Prepended tags appear first; if the model also predicted them they aren't doubled."""
        model = {"img.jpg": {"1girl": 0.95, "smile": 0.7, "blue hair": 0.5}}
        result = merge_model_results(
            [model], prepend_tags=["solo", "1girl"]
        )
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        # "solo" and "1girl" lead; "1girl" not repeated in body
        assert tags[0] == "solo"
        assert tags[1] == "1girl"
        assert tags.count("1girl") == 1
        # remaining body (smile, blue hair) follow
        assert "smile" in tags
        assert "blue hair" in tags

    def test_max_tags_cap(self):
        model = {"img.jpg": {f"tag{i}": (100 - i) / 100 for i in range(50)}}
        result = merge_model_results([model], max_tags=10)
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert len(tags) == 10

    def test_kaomoji_underscores_intact(self):
        """Kaomojis like 0_0 must NOT have underscores replaced."""
        # merge_model_results receives already-decoded tags (replace_underscores
        # merge_model_results receives already-decoded tags; to test the pass-through,
        # we pass a kaomoji tag directly and ensure the merge doesn't corrupt it.
        model = {"img.jpg": {"0_0": 0.8, "long hair": 0.6}}
        result = merge_model_results([model])
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert "0_0" in tags
        assert "long hair" in tags

    def test_underscores_false_converts_to_spaces(self):
        """Default output form collapses underscores to spaces, keeping kaomojis."""
        model = {"img.jpg": {"long_hair": 0.9, "jpeg_artifacts": 0.7, "0_0": 0.5}}
        result = merge_model_results([model])  # underscores defaults to False
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["long hair", "jpeg artifacts", "0_0"]

    def test_underscores_true_keeps_original_form(self):
        """underscores=True preserves the original danbooru form."""
        model = {"img.jpg": {"long_hair": 0.9, "jpeg_artifacts": 0.7}}
        result = merge_model_results([model], underscores=True)
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["long_hair", "jpeg_artifacts"]

    def test_exclude_underscore_insensitive(self):
        """An exclude entry matches regardless of underscore vs space form."""
        model = {"img.jpg": {"long_hair": 0.9, "blue_eyes": 0.7}}
        # exclude written with a space still drops the underscore-form tag
        result = merge_model_results([model], exclude_tags=["long hair"], underscores=True)
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["blue_eyes"]

    def test_single_model_passthrough(self):
        model = {"a.jpg": {"solo": 0.9}, "b.jpg": {"duo": 0.7}}
        result = merge_model_results([model])
        assert result["a.jpg"] == "solo"
        assert result["b.jpg"] == "duo"

    def test_empty_image_yields_empty_string(self):
        model = {"img.jpg": {}}
        result = merge_model_results([model])
        assert result["img.jpg"] == ""

    def test_prepend_only_no_model_tags(self):
        model = {"img.jpg": {}}
        result = merge_model_results([model], prepend_tags=["solo", "masterpiece"])
        tags = [t.strip() for t in result["img.jpg"].split(",")]
        assert tags == ["solo", "masterpiece"]


# ---------------------------------------------------------------------------
# run_ensemble: end-to-end with fake infer_factory
# ---------------------------------------------------------------------------

class TestRunEnsemble:
    """Uses infer_factory to inject fake models — onnxruntime never imported."""

    def test_sequential_model_order_recorded(self, three_tmp_images: list[Path]):
        """Models must be called in spec list order; call log verifies sequence."""
        call_log: list[str] = []

        def _factory(spec: TaggerModelSpec):
            def _infer(batch_paths: list[Path]) -> list[dict[str, float]]:
                call_log.append(spec.id)
                return [{"cat": 0.9} for _ in batch_paths]
            return _infer

        specs = [_make_spec("model-A"), _make_spec("model-B")]
        run_ensemble(three_tmp_images, specs, infer_factory=_factory)

        # model-A must have been called before model-B across all batches
        a_indices = [i for i, s in enumerate(call_log) if s == "model-A"]
        b_indices = [i for i, s in enumerate(call_log) if s == "model-B"]
        assert a_indices, "model-A was never called"
        assert b_indices, "model-B was never called"
        assert max(a_indices) < min(b_indices), (
            "model-B was called before all model-A batches finished"
        )

    def test_merged_caption_lines_returned(self, three_tmp_images: list[Path]):
        """result keys are path strings; values are comma-joined tag lines."""
        def _factory(spec: TaggerModelSpec):
            if spec.id == "model-X":
                def _infer(batch: list[Path]) -> list[dict[str, float]]:
                    return [{"cat": 0.9, "dog": 0.3} for _ in batch]
            else:
                def _infer(batch: list[Path]) -> list[dict[str, float]]:
                    return [{"cat": 0.5, "dog": 0.8, "bird": 0.6} for _ in batch]
            return _infer

        specs = [_make_spec("model-X"), _make_spec("model-Y")]
        result = run_ensemble(three_tmp_images, specs, infer_factory=_factory)

        assert len(result) == 3
        for img in three_tmp_images:
            key = str(img)
            assert key in result
            tags = [t.strip() for t in result[key].split(",")]
            # cat max(0.9,0.5)=0.9, dog max(0.3,0.8)=0.8, bird=0.6 → cat,dog,bird
            assert tags == ["cat", "dog", "bird"]

    def test_on_progress_callback_called(self, three_tmp_images: list[Path]):
        progress_events: list[tuple] = []

        def _progress(done: int, total: int, phase: str) -> None:
            progress_events.append((done, total, phase))

        def _factory(spec: TaggerModelSpec):
            def _infer(batch: list[Path]) -> list[dict[str, float]]:
                return [{"tag": 0.9} for _ in batch]
            return _infer

        specs = [_make_spec("m1")]
        run_ensemble(
            three_tmp_images, specs,
            infer_factory=_factory,
            batch_size=2,
            on_progress=_progress,
        )
        assert len(progress_events) >= 2  # at least one initial + one per-batch call

    def test_exclude_and_prepend_applied(self, three_tmp_images: list[Path]):
        def _factory(spec: TaggerModelSpec):
            def _infer(batch: list[Path]) -> list[dict[str, float]]:
                return [{"cat": 0.9, "nsfw": 0.8, "dog": 0.5} for _ in batch]
            return _infer

        result = run_ensemble(
            three_tmp_images,
            [_make_spec("m")],
            infer_factory=_factory,
            exclude_tags=["nsfw"],
            prepend_tags=["solo"],
        )
        for key, line in result.items():
            tags = [t.strip() for t in line.split(",")]
            assert tags[0] == "solo"
            assert "nsfw" not in tags
            assert "cat" in tags


# ---------------------------------------------------------------------------
# KNOWN_TAGGERS registry sanity
# ---------------------------------------------------------------------------

class TestKnownTaggersRegistry:
    def test_thresholds_in_unit_interval(self):
        for tid, spec in KNOWN_TAGGERS.items():
            assert 0 < spec.general_threshold < 1, (
                f"{tid}: general_threshold {spec.general_threshold} out of (0,1)"
            )
            assert 0 < spec.character_threshold < 1, (
                f"{tid}: character_threshold {spec.character_threshold} out of (0,1)"
            )
            assert 0 < spec.rating_threshold < 1, (
                f"{tid}: rating_threshold {spec.rating_threshold} out of (0,1)"
            )

    def test_required_fields_populated(self):
        for tid, spec in KNOWN_TAGGERS.items():
            assert spec.repo_id, f"{tid}: repo_id empty"
            assert spec.filename, f"{tid}: filename empty"
            assert spec.tags_filename, f"{tid}: tags_filename empty"
            assert spec.input_size > 0, f"{tid}: input_size must be positive"

    def test_known_tagger_ids(self):
        expected = {
            "pixai-v0.9",
            "cl-tagger-1.01",
            "wd-eva02-large-v3",
            "wd-vit-large-v3",
            "wd-swinv2-v3",
        }
        assert expected <= set(KNOWN_TAGGERS.keys()), (
            f"Missing tagger entries: {expected - set(KNOWN_TAGGERS.keys())}"
        )

    def test_cl_tagger_has_subdir_and_json_tags(self):
        spec = KNOWN_TAGGERS["cl-tagger-1.01"]
        assert spec.subdir, "cl-tagger-1.01 must have a non-empty subdir"
        assert spec.tags_filename.endswith(".json"), (
            "cl-tagger-1.01 tags_filename should be a .json mapping"
        )

    def test_wd_taggers_use_csv(self):
        wd_ids = {"wd-eva02-large-v3", "wd-vit-large-v3", "wd-swinv2-v3"}
        for tid in wd_ids:
            spec = KNOWN_TAGGERS[tid]
            assert spec.tags_filename.endswith(".csv"), (
                f"{tid}: expected .csv tags_filename, got {spec.tags_filename}"
            )

    def test_dataclass_is_frozen(self):
        spec = KNOWN_TAGGERS["pixai-v0.9"]
        with pytest.raises((AttributeError, TypeError)):
            spec.id = "mutated"  # type: ignore[misc]


class TestOverrides:
    def test_overrides_replace_spec_fields_before_factory(self, three_tmp_images):
        from rengu_flow.prep.tagger import KNOWN_TAGGERS, run_ensemble

        seen_specs = []

        def factory(spec):
            seen_specs.append(spec)
            return lambda paths: [{} for _ in paths]

        spec = KNOWN_TAGGERS["pixai-v0.9"]
        run_ensemble(
            three_tmp_images,
            [spec],
            overrides={"pixai-v0.9": {"general_threshold": 0.5, "include_rating": False}},
            infer_factory=factory,
        )
        assert seen_specs[0].general_threshold == 0.5
        assert seen_specs[0].include_rating is False
        # The registry entry itself is untouched (frozen dataclass replaced, not mutated).
        assert KNOWN_TAGGERS["pixai-v0.9"].general_threshold == 0.30


class TestPreprocessModes:
    def _img(self, w=300, h=200):
        from PIL import Image

        return Image.new("RGB", (w, h), (255, 0, 0))

    def test_wd_mode_is_nhwc_bgr_0_255(self):
        from rengu_flow.prep.tagger import _preprocess_image

        arr = _preprocess_image(self._img(), 448, "wd")
        assert arr.shape == (1, 448, 448, 3)
        assert arr.max() == 255.0
        # Pure-red RGB image -> BGR channel order puts red last.
        center = arr[0, 224, 224]
        assert center[2] == 255.0 and center[0] == 0.0

    def test_norm05_rgb_is_nchw_normalized(self):
        from rengu_flow.prep.tagger import _preprocess_image

        arr = _preprocess_image(self._img(), 448, "norm05_rgb")
        assert arr.shape == (1, 3, 448, 448)
        assert arr.min() >= -1.0 and arr.max() <= 1.0
        # Red channel first (RGB), fully saturated -> +1.
        assert arr[0, 0, 224, 224] == 1.0 and arr[0, 2, 224, 224] == -1.0

    def test_norm05_bgr_pad_is_nchw_bgr_with_padding(self):
        from rengu_flow.prep.tagger import _preprocess_image

        arr = _preprocess_image(self._img(300, 100), 448, "norm05_bgr_pad")
        assert arr.shape == (1, 3, 448, 448)
        # BGR: blue channel first; red content lands in channel 2.
        assert arr[0, 2, 224, 224] == 1.0
        # Top rows are white padding -> all channels +1.
        assert arr[0, 0, 2, 224] == 1.0 and arr[0, 2, 2, 224] == 1.0

    def test_unknown_mode_raises(self):
        from rengu_flow.prep.tagger import _preprocess_image

        with pytest.raises(ValueError):
            _preprocess_image(self._img(), 448, "bogus")

    def test_registry_preprocess_modes(self):
        from rengu_flow.prep.tagger import KNOWN_TAGGERS

        assert KNOWN_TAGGERS["pixai-v0.9"].preprocess == "norm05_rgb"
        assert KNOWN_TAGGERS["cl-tagger-1.02"].preprocess == "norm05_bgr_pad"
        assert KNOWN_TAGGERS["wd-eva02-large-v3"].preprocess == "wd"


class TestTagMappingJson:
    def test_cl_tagger_layout_named_categories(self, tmp_path):
        import json as _json

        from rengu_flow.prep.tagger import load_tag_list

        mapping = {
            "0": {"tag": "general", "category": "Rating"},
            "1": {"tag": "1girl", "category": "General"},
            "2": {"tag": "hatsune_miku", "category": "Character"},
            "3": {"tag": "vocaloid", "category": "Copyright"},
            "4": {"tag": "masterpiece", "category": "Quality"},
        }
        p = tmp_path / "tag_mapping.json"
        p.write_text(_json.dumps(mapping))
        rows = load_tag_list(p)
        assert rows == [
            ("general", 9),
            ("1girl", 0),
            ("hatsune_miku", 4),
            ("vocaloid", 3),
            ("masterpiece", 0),
        ]

    def test_numeric_key_order_not_insertion_order(self, tmp_path):
        import json as _json

        from rengu_flow.prep.tagger import load_tag_list

        mapping = {"10": {"tag": "later", "category": "General"},
                   "2": {"tag": "early", "category": "General"}}
        p = tmp_path / "tag_mapping.json"
        p.write_text(_json.dumps(mapping))
        assert [name for name, _ in load_tag_list(p)] == ["early", "later"]


# ---------------------------------------------------------------------------
# _predict_decoded: a corrupt/unreadable image (None) is skipped, alignment kept
# ---------------------------------------------------------------------------

def test_predict_decoded_skips_failed_keeps_alignment():
    import numpy as np

    from rengu_flow.prep.tagger import _predict_decoded

    # 3 images; index 1 failed to decode (None).
    decoded = [np.zeros((1, 4)), None, np.zeros((1, 4))]
    calls = {}

    def fake_predict(batch):
        calls["rows"] = batch.shape[0]   # only the 2 valid arrays should be concatenated
        return [{"a": 0.9}, {"b": 0.8}]

    out = _predict_decoded(decoded, fake_predict)
    assert calls["rows"] == 2
    assert out == [{"a": 0.9}, {}, {"b": 0.8}]   # bad image -> empty, others aligned


def test_predict_decoded_all_failed_no_predict_call():
    from rengu_flow.prep.tagger import _predict_decoded

    def fake_predict(batch):  # must not be called when nothing decoded
        raise AssertionError("predict should not run with no valid images")

    assert _predict_decoded([None, None], fake_predict) == [{}, {}]

"""Unit tests for rengu_flow.prep.captioner.

Uses a fake backend — no transformers, no GPU, no torch required.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import pytest

from rengu_flow.prep.captioner import (
    CaptionBackend,
    CaptionerConfig,
    _collapse_to_one_line,
    build_prompt,
    caption_folder,
    list_caption_models,
)
from rengu_flow.prep.caption_store import CaptionStore

pytestmark = pytest.mark.no_ui_db

FIXTURE_JPG = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "smoke_cc0"
    / "images"
    / "gb82_01.jpg"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_collapse_strips_reasoning_think_block():
    """Reasoning VLMs (ToriiGate-0.5) emit <think>…</think> before the caption; keep only the
    caption after the final </think>, and collapse newlines."""
    assert (
        _collapse_to_one_line("draft thoughts.\n</think>\nThe image shows a cat.")
        == "The image shows a cat."
    )
    assert _collapse_to_one_line("A dog runs.\nFast.") == "A dog runs. Fast."
    assert _collapse_to_one_line("<think>reasoning got cut off") == ""


def _make_img_dir(tmp_path: Path, names: list[str]) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    for name in names:
        shutil.copy(FIXTURE_JPG, d / name)
    return d


def _write_txt(img_dir: Path, stem: str, *lines: str) -> None:
    """Write a .txt sidecar for *stem* with the given lines."""
    (img_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------


class FakeBackend(CaptionBackend):
    """Deterministic no-GPU backend that records calls."""

    def __init__(self, response_fn: Optional[Callable[[int], str]] = None) -> None:
        self.loaded = False
        self.unloaded = False
        self.recorded_prompts: list[list[str]] = []  # per-batch
        self._response_fn = response_fn or (lambda idx: f"Fake caption {idx}.")

    def load(self) -> None:
        self.loaded = True

    def caption_batch(self, images, prompts: list[str]) -> list[str]:
        self.recorded_prompts.append(list(prompts))
        return [self._response_fn(i) for i in range(len(prompts))]

    def unload(self) -> None:
        self.unloaded = True


def _make_factory(backend: FakeBackend) -> Callable:
    def factory(config: CaptionerConfig) -> CaptionBackend:
        return backend

    return factory


# ---------------------------------------------------------------------------
# Tests: build_prompt (pure function)
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_joycaption_ignores_tags(self):
        cfg = CaptionerConfig(model="joycaption-beta-one")
        tags = ["1girl", "long hair"]
        prompt = build_prompt(cfg, tags)
        assert "<tags>" not in prompt
        assert "1girl" not in prompt
        # Default preset (training-balanced) applies when no custom prompt is set.
        assert "detailed caption" in prompt

    def test_joycaption_custom_prompt(self):
        cfg = CaptionerConfig(model="joycaption-beta-one", prompt="My custom prompt.")
        prompt = build_prompt(cfg, ["tag1"])
        assert prompt == "My custom prompt."

    def test_toriigate_no_tags(self):
        cfg = CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=True)
        prompt = build_prompt(cfg, [])
        assert "<tags>" not in prompt

    def test_toriigate_with_tags_grounding_enabled(self):
        cfg = CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=True)
        tags = ["1girl", "long hair", "smile"]
        prompt = build_prompt(cfg, tags)
        assert "# Booru tags for the image\n[1girl, long hair, smile]" in prompt
        # Native trained format, not free-form instructions (model card discourages those).
        assert prompt.startswith("# Captioning format:")

    def test_toriigate_grounding_disabled(self):
        cfg = CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=False)
        tags = ["1girl", "long hair"]
        prompt = build_prompt(cfg, tags)
        assert "<tags>" not in prompt

    def test_toriigate_grounding_none_tags(self):
        cfg = CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=True)
        prompt = build_prompt(cfg, None)
        assert "<tags>" not in prompt

    def test_unknown_model_falls_back_to_generic_default(self):
        cfg = CaptionerConfig(model="unknown-model")
        prompt = build_prompt(cfg)
        # Should not crash and returns something
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Tests: caption_folder — basic flow
# ---------------------------------------------------------------------------


class TestCaptionFolderBasic:
    def test_caption_lands_on_line_2_tags_untouched(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl, long hair")  # line 1 = tags

        fb = FakeBackend()
        report = caption_folder(
            img_dir,
            CaptionerConfig(model="joycaption-beta-one"),
            backend_factory=_make_factory(fb),
        )

        assert report["captioned"] == 1
        assert report["skipped"] == 0
        assert report["failed"] == []
        assert not report["stopped"]

        cs = CaptionStore.open(img_dir)
        lines = cs.get_lines("a.jpg")
        assert lines[0] == "1girl, long hair"  # tag line untouched
        assert lines[1].startswith("Fake caption")

    def test_skip_if_line_2_exists_no_overwrite(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg"])
        _write_txt(img_dir, "a", "tags a", "Existing caption.")
        _write_txt(img_dir, "b", "tags b")  # no caption yet

        fb = FakeBackend()
        report = caption_folder(
            img_dir,
            CaptionerConfig(model="joycaption-beta-one", overwrite=False),
            backend_factory=_make_factory(fb),
        )

        assert report["captioned"] == 1   # only b
        assert report["skipped"] == 1     # a was skipped

        cs = CaptionStore.open(img_dir)
        assert cs.get_lines("a.jpg")[1] == "Existing caption."

    def test_overwrite_replaces_existing_caption(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "tags", "Old caption.")

        fb = FakeBackend(lambda _: "New caption.")
        report = caption_folder(
            img_dir,
            CaptionerConfig(model="joycaption-beta-one", overwrite=True),
            backend_factory=_make_factory(fb),
        )

        assert report["captioned"] == 1
        cs = CaptionStore.open(img_dir)
        assert cs.get_lines("a.jpg")[1] == "New caption."

    def test_backend_loaded_and_unloaded(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        fb = FakeBackend()
        caption_folder(
            img_dir,
            CaptionerConfig(),
            backend_factory=_make_factory(fb),
        )
        assert fb.loaded
        assert fb.unloaded

    def test_no_images_no_error(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        fb = FakeBackend()
        report = caption_folder(
            d,
            CaptionerConfig(),
            backend_factory=_make_factory(fb),
        )
        assert report["captioned"] == 0

    def test_list_caption_models_contains_known_models(self):
        models = list_caption_models()
        assert "joycaption-beta-one" in models
        assert "toriigate-0.5" in models


# ---------------------------------------------------------------------------
# Tests: ToriiGate grounding prompt in practice
# ---------------------------------------------------------------------------


class TestToriiGateGrounding:
    def test_grounding_tags_included_in_batch_prompts(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl, smile")

        fb = FakeBackend()
        caption_folder(
            img_dir,
            CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=True),
            backend_factory=_make_factory(fb),
        )

        assert len(fb.recorded_prompts) == 1
        batch_prompts = fb.recorded_prompts[0]
        assert len(batch_prompts) == 1
        assert "# Booru tags for the image\n[1girl, smile]" in batch_prompts[0]

    def test_grounding_disabled_no_tags_in_prompt(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl, smile")

        fb = FakeBackend()
        caption_folder(
            img_dir,
            CaptionerConfig(model="toriigate-0.5", use_tags_as_grounding=False),
            backend_factory=_make_factory(fb),
        )

        batch_prompts = fb.recorded_prompts[0]
        assert "<tags>" not in batch_prompts[0]


# ---------------------------------------------------------------------------
# Tests: incremental save + should_stop
# ---------------------------------------------------------------------------


class TestIncrementalSaveAndStop:
    def test_incremental_save_after_each_batch(self, tmp_path):
        """After each batch, the caption file on disk should be updated."""
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
        # Pre-write tag lines so captions land at index 1
        for stem in ("a", "b", "c"):
            _write_txt(img_dir, stem, "tag1, tag2")

        save_counts: list[int] = []

        class InstrumentedFakeBackend(CaptionBackend):
            def load(self):
                pass

            def caption_batch(self, images, prompts):
                # Count how many txt files exist right before return
                return [f"Caption {i}." for i in range(len(prompts))]

            def unload(self):
                pass

        fb = InstrumentedFakeBackend()

        # batch_size=1 so each image is its own batch
        report = caption_folder(
            img_dir,
            CaptionerConfig(batch_size=1),
            backend_factory=lambda _: fb,
        )

        assert report["captioned"] == 3
        # All three txt files must exist after the run
        for name in ("a.txt", "b.txt", "c.txt"):
            assert (img_dir / name).exists()

    def test_should_stop_triggers_partial_results(self, tmp_path):
        """should_stop after first batch causes early exit; partial .txt written."""
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
        # Pre-write tag lines so the caption lands at index 1
        for stem in ("a", "b", "c"):
            _write_txt(img_dir, stem, "tag1, tag2")

        call_count = 0

        def should_stop():
            nonlocal call_count
            call_count += 1
            # Stop after the first batch check (second call = after processing batch 0)
            return call_count > 1

        fb = FakeBackend()
        report = caption_folder(
            img_dir,
            CaptionerConfig(batch_size=1),
            backend_factory=_make_factory(fb),
            should_stop=should_stop,
        )

        assert report["stopped"] is True
        # At least one image was captioned, but not necessarily all three
        assert report["captioned"] >= 1
        assert report["captioned"] < 3

        # The already-saved captions must be on disk
        captioned_keys = []
        cs = CaptionStore.open(img_dir)
        for key in cs.keys():
            lines = cs.get_lines(key)
            if len(lines) >= 2 and lines[1]:
                captioned_keys.append(key)
        assert len(captioned_keys) == report["captioned"]


# ---------------------------------------------------------------------------
# Tests: multi-line model output collapsed to one line
# ---------------------------------------------------------------------------


class TestMultilineCollapse:
    def test_multiline_model_output_collapsed(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        # Pre-write a tag line so the caption lands at index 1 (second line)
        _write_txt(img_dir, "a", "1girl, smile")

        multiline_response = "First sentence.\nSecond sentence.\nThird sentence."

        class MultilineFakeBackend(CaptionBackend):
            def load(self):
                pass

            def caption_batch(self, images, prompts):
                return [multiline_response] * len(images)

            def unload(self):
                pass

        report = caption_folder(
            img_dir,
            CaptionerConfig(),
            backend_factory=lambda _: MultilineFakeBackend(),
        )

        cs = CaptionStore.open(img_dir)
        lines = cs.get_lines("a.jpg")
        assert lines[0] == "1girl, smile"  # tag line untouched
        # Line 2 (index 1) must be a single line with no embedded newlines
        caption_line = lines[1]
        assert "\n" not in caption_line
        # Content preserved (collapsed)
        assert "First sentence." in caption_line
        assert "Second sentence." in caption_line


# ---------------------------------------------------------------------------
# Tests: on_progress callback
# ---------------------------------------------------------------------------


class TestOnProgress:
    def test_progress_callback_called(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg"])
        progress_calls: list[tuple] = []

        def on_progress(done, total, msg):
            progress_calls.append((done, total, msg))

        fb = FakeBackend()
        caption_folder(
            img_dir,
            CaptionerConfig(batch_size=1),
            backend_factory=_make_factory(fb),
            on_progress=on_progress,
        )

        assert len(progress_calls) == 2
        # Final call: done == total
        last_done, last_total, _ = progress_calls[-1]
        assert last_total == 2


class TestImageSizeNormalization:
    def _setup(self, tmp_path, sizes):
        from PIL import Image as PILImage

        d = tmp_path / "imgs"
        d.mkdir()
        for name, (w, h) in sizes.items():
            PILImage.new("RGB", (w, h), (120, 120, 120)).save(d / name, quality=90)
            (d / name).with_suffix(".txt").write_text("tag line\n")
        return d

    def test_oversized_images_downscaled_before_backend(self, tmp_path):
        from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

        d = self._setup(tmp_path, {"big.jpg": (4096, 2048), "ok.jpg": (800, 600)})
        seen_sizes = {}

        class Backend:
            def load(self):
                pass

            def unload(self):
                pass

            def caption_batch(self, images, prompts):
                for img in images:
                    seen_sizes[img.size] = True
                return ["caption"] * len(images)

        config = CaptionerConfig(max_image_side=1024, batch_size=4)
        report = caption_folder(d, config, backend_factory=lambda c: Backend())
        assert report["captioned"] == 2
        assert (1024, 512) in seen_sizes  # 4096x2048 capped, aspect kept
        assert (800, 600) in seen_sizes  # under the cap: untouched

    def test_too_small_images_skipped_and_reported(self, tmp_path):
        from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

        d = self._setup(tmp_path, {"tiny.jpg": (120, 90), "ok.jpg": (800, 600)})

        class Backend:
            def load(self):
                pass

            def unload(self):
                pass

            def caption_batch(self, images, prompts):
                return ["caption"] * len(images)

        config = CaptionerConfig(min_image_side=256, max_image_side=0, batch_size=4)
        report = caption_folder(d, config, backend_factory=lambda c: Backend())
        assert report["captioned"] == 1
        assert report["skipped_small"] == ["tiny.jpg"]
        # The tiny image's caption file is untouched (still only the tag line).
        assert (d / "tiny.txt").read_text() == "tag line\n"


class TestComposablePrompts:
    def test_compose_default_matches_training_balanced_intent(self):
        from rengu_flow.prep.captioner import compose_prompt

        text = compose_prompt()
        assert "long, detailed caption" in text
        assert "apparent age" in text and "ethnicity" in text  # demographics default
        assert "avoid useless" in text  # no-meta always appended last

    def test_medium_neutral_modifier_stacks(self):
        from rengu_flow.prep.captioner import compose_prompt

        text = compose_prompt(modifiers=["demographics", "medium_neutral"])
        lower = text.lower()
        assert "never mention or hint at the medium" in lower
        for word in ("photo", "anime", "illustration", "render", "realistic"):
            assert word in lower  # listed as forbidden words in the instruction
        assert "apparent age" in text

    def test_character_trigger_and_outfit_policies(self):
        from rengu_flow.prep.captioner import compose_prompt

        described = compose_prompt(
            base="character-focus", character_name="hatsune miku", outfit="describe"
        )
        assert "refer to them as 'hatsune miku'" in described
        assert "never describe hatsune miku's unchangeable physical traits" in described
        # The trait prohibition is the LAST constraint (recency wins with VLMs).
        assert described.rindex("unchangeable") > described.rindex("explicitly and in detail")
        assert "explicitly and in detail" in described  # outfit describable

        omitted = compose_prompt(
            base="character-focus", character_name="hatsune miku", outfit="omit"
        )
        assert "leave it completely unmentioned" in omitted

        # Without a character name the outfit policy adds nothing.
        plain = compose_prompt(outfit="omit")
        assert "unmentioned" not in plain and "refer to them" not in plain

    def test_outfit_mixed_is_deterministic_and_split(self):
        from rengu_flow.prep.captioner import compose_prompt

        keys = [f"img_{i}.jpg" for i in range(40)]
        sides = {
            key: "explicitly and in detail"
            in compose_prompt(character_name="x", outfit="mixed", image_key=key)
            for key in keys
        }
        # Deterministic: same key, same side.
        for key in keys[:5]:
            assert (
                "explicitly and in detail"
                in compose_prompt(character_name="x", outfit="mixed", image_key=key)
            ) == sides[key]
        # Roughly split: both sides present in a 40-image set.
        assert 5 < sum(sides.values()) < 35

    def test_validation_and_options_listing(self):
        from rengu_flow.prep.captioner import compose_prompt, list_prompt_options

        with pytest.raises(ValueError):
            compose_prompt(base="nope")
        with pytest.raises(ValueError):
            compose_prompt(modifiers=["nope"])
        with pytest.raises(ValueError):
            compose_prompt(outfit="naked")

        options = list_prompt_options()
        assert {b["id"] for b in options["bases"]} >= {
            "descriptive-long", "concise", "character-focus", "style-focus"
        }
        assert {m["id"] for m in options["modifiers"]} >= {
            "demographics",
            "medium_neutral",
            "plain_language",
            "objective_only",
            "composition_camera",
            "explicit_language",
        }
        assert options["outfit_modes"] == ["describe", "omit", "mixed"]

    def test_custom_prompt_overrides_composition(self):
        from rengu_flow.prep.captioner import CaptionerConfig, build_prompt

        config = CaptionerConfig(prompt="My custom.", character_name="miku")
        assert build_prompt(config) == "My custom."

    def test_toriigate_native_prompt_structure(self):
        from rengu_flow.prep.captioner import CaptionerConfig, build_prompt

        config = CaptionerConfig(
            model="toriigate-0.5",
            prompt_modifiers=("medium_neutral",),
            character_name="hatsune miku",
            outfit="omit",
        )
        text = build_prompt(config, tags=["1girl", "long hair"])
        # Official trained blocks, in order: format, extra requirements, tags, characters.
        assert text.startswith("# Captioning format:")
        assert "Use 2 to 5 paragraphs" in text  # the 'long' trained format, verbatim
        assert "# Extra requirements:" in text
        assert "never mention or hint at the medium" in text.lower()
        assert "leave it completely unmentioned" in text  # outfit omit
        assert "# Booru tags for the image\n[1girl, long hair]" in text
        assert "make sure to use them: [hatsune miku]" in text
        # Custom prompt still wins and grounds with the official block.
        custom = CaptionerConfig(model="toriigate-0.5", prompt="My custom.")
        text = build_prompt(custom, tags=["1girl"])
        assert text.startswith("My custom.")
        assert "# Booru tags for the image\n[1girl]" in text

    def test_toriigate_concise_maps_to_short_format(self):
        from rengu_flow.prep.captioner import CaptionerConfig, build_prompt

        config = CaptionerConfig(model="toriigate-0.5", prompt_base="concise")
        text = build_prompt(config)
        assert "quite short without long purple prose" in text


    def test_register_modifiers_compose(self):
        from rengu_flow.prep.captioner import compose_prompt

        text = compose_prompt(
            modifiers=["plain_language", "objective_only", "composition_camera"]
        )
        assert "simple, plain English" in text
        assert "never evaluate" in text
        assert "shot type" in text


class TestTraitScrubber:
    def test_scrubs_trait_clauses_keeps_rest(self):
        from rengu_flow.prep.captioner import scrub_trait_clauses

        text = (
            "Saki stands against a white background, smiling at the camera. "
            "She has long brown hair and is wearing a pink headband. "
            "Saki is holding a black smartphone in her right hand."
        )
        out = scrub_trait_clauses(text)
        assert "brown hair" not in out
        assert "pink headband" in out  # clothing clause in the same sentence survives
        assert "white background" in out and "smartphone" in out

    def test_scrubs_eye_color_age_and_build(self):
        from rengu_flow.prep.captioner import scrub_trait_clauses

        out = scrub_trait_clauses(
            "Miku smiles warmly. She has blue eyes. She appears to be in her "
            "early twenties. Her slender figure leans against the wall. "
            "She wears a school uniform."
        )
        assert "blue eyes" not in out
        assert "twenties" not in out
        assert "slender" not in out
        assert "school uniform" in out and "smiles warmly" in out

    def test_clean_captions_pass_through(self):
        from rengu_flow.prep.captioner import scrub_trait_clauses

        text = "Saki is sitting on a bench in a park, holding an umbrella."
        assert scrub_trait_clauses(text) == text

    def test_caption_folder_scrubs_only_with_trigger(self, tmp_path):
        from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "tags")
        leaky = FakeBackend(lambda _: "Saki smiles. She has long brown hair.")

        report = caption_folder(
            img_dir,
            CaptionerConfig(character_name="Saki", overwrite=True),
            backend_factory=_make_factory(leaky),
        )
        assert report["captioned"] == 1
        line2 = (img_dir / "a.txt").read_text().splitlines()[1]
        assert "brown hair" not in line2 and "Saki smiles." in line2

        # Without a trigger the caption is untouched.
        caption_folder(
            img_dir,
            CaptionerConfig(overwrite=True),
            backend_factory=_make_factory(leaky),
        )
        line2 = (img_dir / "a.txt").read_text().splitlines()[1]
        assert "brown hair" in line2


class TestCharacterVariantsAndTargetLine:
    def test_canon_mode_prompt_describes_deviations(self):
        from rengu_flow.prep.captioner import compose_prompt

        text = compose_prompt(
            character_name="hatsune miku",
            character_canon="aqua twin-tail hair, blue eyes, slim teenage build",
        )
        assert "canonical look is: aqua twin-tail hair" in text
        assert "deviates from the canonical look" in text
        assert "describe that deviation explicitly" in text
        # Plain trigger text is replaced, not stacked.
        assert "Wherever you would normally describe" not in text

    def test_canon_mode_disables_scrubber(self, tmp_path):
        from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "tags")
        deviation = FakeBackend(
            lambda _: "miku appears aged-up with short pink hair instead of her usual look."
        )
        caption_folder(
            img_dir,
            CaptionerConfig(
                character_name="miku",
                character_canon="aqua twintails, blue eyes",
                overwrite=True,
            ),
            backend_factory=_make_factory(deviation),
        )
        line2 = (img_dir / "a.txt").read_text().splitlines()[1]
        assert "short pink hair" in line2  # deviation survived (no scrubbing)

    def test_target_line_3_adds_a_variant(self, tmp_path):
        from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "tags", "Absorbed caption.")

        fb = FakeBackend(lambda _: "Full description caption.")
        report = caption_folder(
            img_dir,
            CaptionerConfig(target_line=3),
            backend_factory=_make_factory(fb),
        )
        assert report["captioned"] == 1
        lines = (img_dir / "a.txt").read_text().splitlines()
        assert lines == ["tags", "Absorbed caption.", "Full description caption."]

        # Re-run without overwrite: line 3 exists -> skipped.
        report = caption_folder(
            img_dir,
            CaptionerConfig(target_line=3),
            backend_factory=_make_factory(fb),
        )
        assert report["skipped"] == 1 and report["captioned"] == 0


class TestSamplingDefaultsAndExactMode:
    def test_model_default_sampling_resolution(self):
        from rengu_flow.prep.captioner import _BACKENDS

        assert _BACKENDS["toriigate-0.5"]["default_temperature"] == 0.5
        assert _BACKENDS["toriigate-0.5"]["default_top_p"] == 1.0
        assert _BACKENDS["joycaption-beta-one"]["default_temperature"] == 0.6

    def test_exact_generation_routes_per_image(self):
        from rengu_flow.prep.captioner import CaptionerConfig, ToriiGateBackend

        calls = []

        class Probe(ToriiGateBackend):
            def _generate(self, images, prompts):
                calls.append(len(images))
                return ["cap"] * len(images)

            def _prepare_image(self, image):
                return image

        def armed(b):
            b._processor = object()
            b._model = object()
            return b

        # exact off (default): one batched call.
        backend = armed(Probe(CaptionerConfig(model="toriigate-0.5")))
        backend.caption_batch([object()] * 3, ["p"] * 3)
        assert calls == [3]
        calls.clear()
        # exact on: one call per image.
        backend = armed(Probe(CaptionerConfig(model="toriigate-0.5", exact_generation=True)))
        backend.caption_batch([object()] * 3, ["p"] * 3)
        assert calls == [1, 1, 1]
        calls.clear()
        # JoyCaption ignores the knob (padding-safe full attention).
        from rengu_flow.prep.captioner import JoyCaptionBackend

        class JoyProbe(JoyCaptionBackend):
            def _generate(self, images, prompts):
                calls.append(len(images))
                return ["cap"] * len(images)

            def _prepare_image(self, image):
                return image

        backend = armed(JoyProbe(CaptionerConfig(exact_generation=True)))
        backend.caption_batch([object()] * 3, ["p"] * 3)
        assert calls == [3]


# ---------------------------------------------------------------------------
# Tests: vLLM overlay path (subprocess mocked — no vLLM, no GPU)
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for the vLLM overlay: reads the manifest argv and emits one RESULT
    line per item, exactly as vllm_captioner.py would, with returncode 0."""

    def __init__(self, cmd, stdout=None, stderr=None, text=None, env=None):
        import io
        import json as _json

        from rengu_flow.prep.vllm_captioner import RESULT_PREFIX

        manifest = _json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        lines = [
            RESULT_PREFIX + _json.dumps({"key": it["key"], "caption": f"vllm caption for {it['key']}"}) + "\n"
            for it in manifest["items"]
        ]
        # interleave a noise line the parent must ignore
        self.stdout = io.StringIO("INFO vllm starting...\n" + "".join(lines))
        self.returncode = 0

    def wait(self):
        return 0

    def terminate(self):
        self.returncode = -15


class TestCaptionFolderVLLM:
    def test_vllm_engine_writes_captions(self, tmp_path, monkeypatch):
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg"])
        _write_txt(img_dir, "a", "1girl")
        _write_txt(img_dir, "b", "1boy")

        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        report = caption_folder(
            img_dir,
            CaptionerConfig(model="joycaption-beta-one", engine="vllm"),
        )
        assert report["captioned"] == 2
        assert not report["failed"]
        cs = CaptionStore.open(img_dir, fmt="sidecar", ext=".txt")
        for key in cs.keys():
            assert cs.get_lines(key)[1] == f"vllm caption for {key}"

    def test_vllm_rejects_non_joycaption(self, tmp_path, monkeypatch):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl")
        monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
        with pytest.raises(ValueError, match="joycaption"):
            caption_folder(img_dir, CaptionerConfig(model="toriigate-0.5", engine="vllm"))


# ---------------------------------------------------------------------------
# Tests: GGUF (llama.cpp) engine — server + downloads mocked, no GPU/binary/net
# ---------------------------------------------------------------------------


class _FakeServerProc:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.returncode = -9


class TestCaptionFolderGGUF:
    def _mock_gg(self, monkeypatch, tmp_path, caption_fn):
        import rengu_flow.prep.gguf_captioner as gg

        monkeypatch.setattr(gg, "ensure_binary", lambda: tmp_path)
        monkeypatch.setattr(gg, "ensure_gguf", lambda q: (tmp_path / "m.gguf", tmp_path / "mm.gguf"))
        monkeypatch.setattr(gg, "_start_server", lambda *a, **k: _FakeServerProc())
        monkeypatch.setattr(gg, "_wait_health", lambda *a, **k: None)
        monkeypatch.setattr(gg, "_encode_image", lambda p: "b64")
        monkeypatch.setattr(gg, "_request_caption", caption_fn)

    def test_gguf_engine_writes_captions(self, tmp_path, monkeypatch):
        img_dir = _make_img_dir(tmp_path, ["a.jpg", "b.jpg"])
        _write_txt(img_dir, "a", "1girl")
        _write_txt(img_dir, "b", "1boy")
        self._mock_gg(monkeypatch, tmp_path, lambda port, b64, prompt, cfg: "gguf caption here")

        report = caption_folder(
            img_dir, CaptionerConfig(model="toriigate-0.5", engine="gguf", gguf_quantization="Q5_K_M")
        )
        assert report["captioned"] == 2
        assert not report["failed"]
        cs = CaptionStore.open(img_dir, fmt="sidecar", ext=".txt")
        for key in cs.keys():
            assert cs.get_lines(key)[1] == "gguf caption here"

    def test_gguf_request_failure_marks_failed_not_crash(self, tmp_path, monkeypatch):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl")

        def boom(port, b64, prompt, cfg):
            raise RuntimeError("server 500")

        self._mock_gg(monkeypatch, tmp_path, boom)
        report = caption_folder(img_dir, CaptionerConfig(model="toriigate-0.5", engine="gguf"))
        assert report["captioned"] == 0
        assert report["failed"] == ["a.jpg"]

    def test_gguf_rejects_non_toriigate(self, tmp_path):
        img_dir = _make_img_dir(tmp_path, ["a.jpg"])
        _write_txt(img_dir, "a", "1girl")
        with pytest.raises(ValueError, match="toriigate"):
            caption_folder(img_dir, CaptionerConfig(model="joycaption-beta-one", engine="gguf"))


def test_gguf_release_asset_per_platform(monkeypatch):
    """_release_asset must not touch a non-existent Platform.is_macos (regression)."""
    import rengu_flow.prep.gguf_captioner as gg

    class _Win:
        is_windows = True

    class _Posix:
        is_windows = False

    monkeypatch.setattr("rengu_flow.platform_compat.PLATFORM", _Posix())
    monkeypatch.setattr("sys.platform", "linux")
    assert gg._release_asset().endswith("ubuntu-vulkan-x64.tar.gz")

    monkeypatch.setattr("rengu_flow.platform_compat.PLATFORM", _Win())
    assert gg._release_asset().endswith("win-vulkan-x64.zip")

    monkeypatch.setattr("rengu_flow.platform_compat.PLATFORM", _Posix())
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="macOS"):
        gg._release_asset()

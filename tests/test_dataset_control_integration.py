"""Edit datasets end to end on the data side — no test doubles for the data layer.

A real ``DatasetManager`` caches a real ``Dataset`` built from a temporary folder tree with one
text-to-image directory and two edit directories (one control per target, and two controls per
target with their own aspect ratios). The only stand-in is the model: a minimal pipeline stub that
honours the data/model contract (VAE fn -> ``control_latents_i`` shaped from its input; 3-argument
text-encoder fn that records what it receives). Checks:

* batches reaching ``prepare_inputs`` are homogeneous (never t2i mixed with edit, never different
  control counts/sizes) and carry the right keys and shapes;
* control sizes come from ``rengu_flow.data.control`` (own aspect ratio, not the target bucket);
* replacing a control file invalidates both the text-embedding and the latent cache for that row.
"""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from rengu_flow.data.control import control_size
from rengu_flow.data.dataset import Dataset
from rengu_flow.data.manager import DatasetManager
from rengu_flow.data.preprocess_media import PreprocessMediaFile
from rengu_flow.engine import select_backend

pytestmark = pytest.mark.no_ui_db

RES = 64
MULTIPLE = 16
LATENT_CH = 4


class StubPipeline:
    """Smallest model that satisfies the DatasetManager/Dataset contract for edit training."""

    name = "stub_edit"
    pixels_round_to_multiple = MULTIPLE
    framerate = None

    def __init__(self):
        self.vae = SimpleNamespace()
        self.text_encoder = SimpleNamespace()
        self.vae_calls: list[dict] = []
        self.te_calls: list[dict] = []
        self.prepared: list[dict] = []

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def get_preprocess_media_file_fn(self, augmentation_resolver=None):
        return PreprocessMediaFile({}, support_video=False, augmentation_resolver=augmentation_resolver)

    def get_call_vae_fn(self, vae):
        def fn(tensor, control_tensors=None):
            call = {"target": tuple(tensor.shape), "controls": None}
            b, _c, h, w = tensor.shape
            out = {"latents": tensor.new_full((b, LATENT_CH, h // 8, w // 8), float(tensor.mean()))}
            if control_tensors is not None:
                call["controls"] = [tuple(t.shape) for t in control_tensors]
                call["control_means"] = [float(t.mean()) for t in control_tensors]
                for i, t in enumerate(control_tensors):
                    cb, _cc, cf, ch, cw = t.shape
                    out[f"control_latents_{i}"] = t.new_full(
                        (cb, LATENT_CH, cf, ch // 8, cw // 8), float(t.mean())
                    )
            self.vae_calls.append(call)
            return out

        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video, control_images):
            self.te_calls.append(
                {
                    "captions": list(captions),
                    "control_images": [
                        None if imgs is None else [(im.size, im.mode) for im in imgs]
                        for imgs in control_images
                    ],
                }
            )
            n = len(captions)
            n_images = [0 if imgs is None else len(imgs) for imgs in control_images]
            return {
                "prompt_embeds": torch.stack(
                    [torch.full((3, 8), float(k)) for k in n_images]
                ).reshape(n, 3, 8),
            }

        return fn

    def keep_submodel_on_cpu_after_cache(self, submodel):
        return True

    def prepare_inputs(self, inputs, timestep_quantile=None):
        self.prepared.append(inputs)
        return (inputs["latents"],), (inputs["latents"], inputs["mask"])


def _img(path, size, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _caption(path, text):
    path.write_text(text, encoding="utf-8")


def _build_tree(root):
    # text-to-image: two square targets, no controls
    for stem, color in (("a", (200, 0, 0)), ("b", (0, 200, 0))):
        _img(root / "t2i" / f"{stem}.png", (64, 64), color)
        _caption(root / "t2i" / f"{stem}.txt", f"t2i {stem}")
    # edit N=1: two wide controls and two tall ones -> two control signatures
    for stem, csize in (("p", (128, 64)), ("q", (128, 64)), ("r", (64, 128)), ("s", (64, 128))):
        _img(root / "edit1" / "targets" / f"{stem}.png", (64, 64), (10, 10, 10))
        _caption(root / "edit1" / "targets" / f"{stem}.txt", f"make {stem} red")
        _img(root / "edit1" / "controls" / f"{stem}.png", csize, (50, 60, 70))
    # edit N=2: each target has a square control and a 3:2 one
    for stem in ("u", "v"):
        _img(root / "edit2" / "targets" / f"{stem}.png", (64, 64), (20, 20, 20))
        _caption(root / "edit2" / "targets" / f"{stem}.txt", f"merge {stem}")
        _img(root / "edit2" / "controls" / f"{stem}_0.png", (64, 64), (1, 2, 3))
        _img(root / "edit2" / "controls" / f"{stem}_1.png", (96, 64), (4, 5, 6))


def _dataset_config(root):
    return {
        "resolutions": [RES],
        "enable_ar_bucket": False,
        "directory": [
            {"path": str(root / "t2i"), "num_repeats": 1},
            {
                "path": str(root / "edit1" / "targets"),
                "control_path": str(root / "edit1" / "controls"),
                "num_repeats": 1,
            },
            {
                "path": str(root / "edit2" / "targets"),
                "control_path": str(root / "edit2" / "controls"),
                "num_repeats": 1,
            },
        ],
    }


def _cache_and_batches(root, edit_dir_overrides=None, **dataset_overrides):
    # Windows cannot overwrite an Arrow file that a not-yet-collected dataset of a previous run
    # still memory-maps; a real launch is a fresh process, so collect the previous run first.
    gc.collect()
    model = StubPipeline()
    training_config = {"cache_root": str(root / "cache")}
    cfg = {**_dataset_config(root), **dataset_overrides}
    for d in cfg["directory"]:
        if "control_path" in d:
            d.update(edit_dir_overrides or {})
    ds = Dataset(cfg, model, training_config=training_config)
    manager = DatasetManager(model, backend=select_backend({"engine": "accelerate"}))
    manager.register(ds)
    manager.cache(unload_models=False)
    ds.post_init(0, 1, {None: 2}, 1, {None: 2})
    for i in range(len(ds)):
        model.prepare_inputs(ds[i])
    return model, ds


WIDE = control_size(128, 64, RES, MULTIPLE)
TALL = control_size(64, 128, RES, MULTIPLE)
SQUARE = control_size(64, 64, RES, MULTIPLE)
THREE_TWO = control_size(96, 64, RES, MULTIPLE)


def test_helper_sizes_differ_from_the_target_bucket():
    # Sanity of the fixture: the controls keep their own aspect, so none equals the 64x64 bucket
    # except the square one.
    assert WIDE == (80, 32) and TALL == (32, 80) and THREE_TWO == (64, 48) and SQUARE == (64, 64)


def test_batches_are_homogeneous_with_controls_from_the_helper(tmp_path):
    _build_tree(tmp_path)
    model, _ds = _cache_and_batches(tmp_path)

    signatures = []
    for batch in model.prepared:
        assert batch["latents"].shape == (2, LATENT_CH, 8, 8)
        control_keys = sorted(k for k in batch if k.startswith("control_latents_"))
        captions = batch["caption"]
        if not control_keys:
            assert all(c.startswith("t2i") for c in captions), captions
            signatures.append(())
            continue
        # Stacked tensors (never ragged lists): homogeneity is what makes torch.stack possible.
        sig = []
        for k in control_keys:
            t = batch[k]
            assert torch.is_tensor(t), f"{k} was not stacked: batch mixed control shapes"
            b, ch, f, h, w = t.shape
            assert (b, ch, f) == (2, LATENT_CH, 1)
            sig.append((w * 8, h * 8))
        signatures.append(tuple(sig))
        if len(control_keys) == 1:
            assert all(c.startswith("make") for c in captions), captions
        else:
            assert all(c.startswith("merge") for c in captions), captions

    assert sorted(signatures) == sorted([(), (WIDE,), (TALL,), (SQUARE, THREE_TWO)])

    # VAE: t2i called with one argument; edit with a list of (B, C, 1, H_i, W_i) at helper sizes.
    t2i_calls = [c for c in model.vae_calls if c["controls"] is None]
    edit_calls = [c for c in model.vae_calls if c["controls"] is not None]
    assert len(t2i_calls) == 2 and len(edit_calls) == 6
    for c in edit_calls:
        for (b, ch, f, h, w) in c["controls"]:
            assert (b, ch, f) == (1, 3, 1)
        sizes = tuple((w, h) for (_b, _c, _f, h, w) in c["controls"])
        assert sizes in {(WIDE,), (TALL,), (SQUARE, THREE_TWO)}

    # TE: control images are PIL at the helper size (or None for t2i and the uncond row).
    seen = {}
    for call in model.te_calls:
        for cap, imgs in zip(call["captions"], call["control_images"]):
            seen[cap] = imgs
    assert seen["t2i a"] is None and seen[""] is None
    assert seen["make p red"] == [(WIDE, "RGB")]
    assert seen["make r red"] == [(TALL, "RGB")]
    assert seen["merge u"] == [(SQUARE, "RGB"), (THREE_TWO, "RGB")]


def test_replacing_a_control_invalidates_te_and_latent_caches(tmp_path):
    _build_tree(tmp_path)
    _cache_and_batches(tmp_path)

    # Same run again: nothing re-encoded.
    again, _ = _cache_and_batches(tmp_path)
    assert again.vae_calls == []
    assert again.te_calls == []

    del again, _  # a relaunch starts from a fresh process (see _cache_and_batches)

    # Replace one control (same size, new pixels, new mtime).
    ctrl = tmp_path / "edit2" / "controls" / "u_1.png"
    _img(ctrl, (96, 64), (250, 250, 250))
    st = ctrl.stat()
    os.utime(ctrl, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    model, _ = _cache_and_batches(tmp_path)
    # Latents: exactly the edited row, with the new control pixels.
    assert len(model.vae_calls) == 1, model.vae_calls
    call = model.vae_calls[0]
    assert [s[-2:] for s in call["controls"]] == [(64, 64), (48, 64)]
    assert call["control_means"][1] == pytest.approx(250 / 127.5 - 1, abs=1e-3)
    # Text embeddings: exactly the edited row's caption.
    encoded = [cap for call in model.te_calls for cap in call["captions"] if cap]
    assert encoded == ["merge u"], encoded


@pytest.mark.parametrize(
    "overrides, expected",
    [
        # Default: each bucket resizes its controls at its own resolution (32 and 64 here), and the
        # text encoder sees the same per-bucket size (its cache is per size bucket for edit rows).
        ({}, {control_size(128, 64, 32, MULTIPLE), control_size(128, 64, 64, MULTIPLE)}),
        # Explicit control_resolution on the [[directory]]: one size for every bucket.
        ({"control_resolution": 48}, {control_size(128, 64, 48, MULTIPLE)}),
    ],
)
def test_control_resolution_default_follows_each_bucket(tmp_path, overrides, expected):
    _build_tree(tmp_path)
    model, ds = _cache_and_batches(tmp_path, edit_dir_overrides=overrides, resolutions=[32, RES])
    te_sizes = {
        imgs[0][0]
        for call in model.te_calls
        for cap, imgs in zip(call["captions"], call["control_images"])
        if cap == "make p red"
    }
    assert te_sizes == expected
    vae_sizes = {
        (c["controls"][0][-1], c["controls"][0][-2])
        for c in model.vae_calls
        if c["controls"] is not None and len(c["controls"]) == 1 and c["controls"][0][-1] > c["controls"][0][-2]
    }
    assert vae_sizes == expected


def test_dropping_control_path_never_reuses_edit_text_embeddings(tmp_path):
    """The edit caches (per size bucket, ``cache_*_ctl_*``) sit next to the text-to-image ones and
    are offered as sibling donors. A t2i row's identity (caption, image) matches the edit row of
    the same target, so without a filter a directory whose ``control_path`` was removed trained
    on embeddings encoded WITH its old control images."""
    _build_tree(tmp_path)
    d = {"path": str(tmp_path / "edit1" / "targets"), "num_repeats": 1}

    def run(with_control):
        gc.collect()
        model = StubPipeline()
        directory = {**d, "control_path": str(tmp_path / "edit1" / "controls")} if with_control else dict(d)
        cfg = {"resolutions": [RES], "enable_ar_bucket": False, "directory": [directory]}
        ds = Dataset(cfg, model, training_config={"cache_root": str(tmp_path / "cache")})
        manager = DatasetManager(model, backend=select_backend({"engine": "accelerate"}))
        manager.register(ds)
        manager.cache(unload_models=False)
        ds.post_init(0, 1, {None: 2}, 1, {None: 2})
        return model, [ds[i] for i in range(len(ds))]

    run(True)
    model, batches = run(False)
    encoded = sorted(cap for call in model.te_calls for cap in call["captions"] if cap)
    assert encoded == ["make p red", "make q red", "make r red", "make s red"], encoded
    for batch in batches:
        assert not any(k.startswith("control_latents_") for k in batch)
        # The stub fills each embedding with its row's control-image count: 0 for t2i.
        assert all(float(e.abs().max()) == 0.0 for e in batch["prompt_embeds"])


def test_corrupt_control_is_tombstoned_on_both_passes(tmp_path):
    """A control whose header reads but whose pixels do not (truncated file) tombstones its row
    in the latent pass; the text-embedding pass must not abort the whole caching run on it."""
    _build_tree(tmp_path)
    ctrl = tmp_path / "edit1" / "controls" / "p.png"
    ctrl.write_bytes(ctrl.read_bytes()[:60])  # PNG header + IHDR survive, the pixel data does not
    model, _ds = _cache_and_batches(tmp_path)
    captions = [c for batch in model.prepared for c in batch["caption"]]
    assert "make p red" not in captions
    # q is p's only bucket-mate (a lone row does not fill a batch of 2); the other bucket trains.
    assert {"make r red", "make s red"} <= set(captions)


class ValidatingStub(StubPipeline):
    """Stub with the optional ``validate_control_rows`` hook: records what it receives."""

    def __init__(self, error=None):
        super().__init__()
        self.validated: list[list] = []
        self.error = error

    def validate_control_rows(self, rows):
        self.validated.append(list(rows))
        if self.error:
            raise ValueError(self.error)


def _cache_with(root, model, **dataset_overrides):
    gc.collect()
    ds = Dataset({**_dataset_config(root), **dataset_overrides}, model,
                 training_config={"cache_root": str(root / "cache")})
    manager = DatasetManager(model, backend=select_backend({"engine": "accelerate"}))
    manager.register(ds)
    manager.cache(unload_models=False)
    return ds


@pytest.mark.parametrize("ar_bucket", [False, True])
def test_validate_control_rows_sees_final_sizes_of_every_edit_row(tmp_path, ar_bucket):
    _build_tree(tmp_path)
    model = ValidatingStub()
    _cache_with(tmp_path, model, resolutions=[32, RES], enable_ar_bucket=ar_bucket)
    assert len(model.validated) == 1  # once, after the metadata stage
    seen = {(os.path.basename(r.target), r.sizes) for r in model.validated[0]}
    for r in model.validated[0]:
        assert r.count == len(r.sizes)
    expected = set()
    for res in (32, RES):  # default control_resolution: each bucket's resolution
        for stem, src in (("p", (128, 64)), ("q", (128, 64)), ("r", (64, 128)), ("s", (64, 128))):
            expected.add((f"{stem}.png", (control_size(*src, res, MULTIPLE),)))
        for stem in ("u", "v"):
            expected.add((f"{stem}.png", (control_size(64, 64, res, MULTIPLE), control_size(96, 64, res, MULTIPLE))))
    assert seen == expected  # t2i rows are never passed


def test_validate_control_rows_failure_stops_before_any_encode(tmp_path):
    _build_tree(tmp_path)
    model = ValidatingStub(error="controls too small")
    with pytest.raises(ValueError, match="controls too small"):
        _cache_with(tmp_path, model)
    assert model.vae_calls == [] and model.te_calls == []

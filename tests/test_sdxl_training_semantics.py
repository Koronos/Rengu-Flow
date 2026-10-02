"""SDXL training semantics: debiased-loss switch, noise scheduler (v-pred / zero-terminal SNR),
preview clip_skip, and the text-cache key (clip_skip, encoder-file identity, encoding version)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("diffusers", exc_type=ImportError)

from rengu_flow.model.sdxl import (
    SDXLPipeline,
    build_noise_scheduler,
    fix_noise_scheduler_betas_for_zero_terminal_snr,
    prepare_scheduler_for_custom_training,
)
from rengu_flow.registry.model_capabilities import get_capability

pytestmark = pytest.mark.no_ui_db


def _snr(scheduler):
    ac = scheduler.alphas_cumprod
    return (ac.sqrt() / (1.0 - ac).sqrt()) ** 2


# ---- debiased_estimation_loss = false must not weight ---------------------------------------


def _loss(flag):
    sched = build_noise_scheduler()
    p = SDXLPipeline.__new__(SDXLPipeline)
    p.__dict__.update(
        config={}, model_config={}, v_pred=False, min_snr_gamma=None,
        debiased_estimation_loss=flag, _pipeline=SimpleNamespace(scheduler=sched),
    )
    torch.manual_seed(0)
    out, tgt = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    return float(p.get_loss_fn()((out, torch.tensor([5, 900])), (tgt, torch.tensor([]))))


def test_debiased_false_is_the_same_as_unset_and_true_weights():
    assert _loss(False) == _loss(None)
    assert _loss(True) != _loss(None)


# ---- scheduler -------------------------------------------------------------------------------


def test_default_scheduler_is_epsilon_without_rescale():
    s = build_noise_scheduler()
    assert s.config.prediction_type == "epsilon" and s.alphas_cumprod[-1] > 0
    assert s.config.timestep_spacing == "leading"


def test_v_pred_scheduler_predicts_velocity_and_has_zero_terminal_snr():
    s = build_noise_scheduler(v_pred=True)
    assert s.config.prediction_type == "v_prediction"  # previews decode the UNet output as velocity
    assert s.config.timestep_spacing == "trailing"
    assert s.alphas_cumprod[-1] == pytest.approx(0.0, abs=1e-12)


def test_all_snr_is_computed_after_the_zero_terminal_rescale():
    s = build_noise_scheduler(v_pred=True)
    torch.testing.assert_close(s.all_snr, _snr(s))
    assert s.all_snr[-1] == pytest.approx(0.0, abs=1e-6)  # the stale (pre-rescale) table ended at ~0.0047


def test_fix_refreshes_an_existing_snr_table():
    s = build_noise_scheduler()
    stale = s.all_snr.clone()
    fix_noise_scheduler_betas_for_zero_terminal_snr(s)
    assert not torch.allclose(s.all_snr, stale)
    torch.testing.assert_close(s.all_snr, _snr(s))


@pytest.mark.parametrize(
    "model_extra,zero_terminal",
    [({}, False), ({"v_pred": True}, True), ({"v_pred": True, "zero_terminal_snr": False}, False),
     ({"zero_terminal_snr": True}, True)],
)
def test_zero_terminal_snr_switch_defaults_to_v_pred(model_extra, zero_terminal):
    cfg = {"model": {"type": "sdxl", "dtype": "float32", "checkpoint_path": "x", **model_extra}}
    m = SDXLPipeline(cfg)
    assert m.zero_terminal_snr is zero_terminal
    s = build_noise_scheduler(m.v_pred, m.zero_terminal_snr)
    assert bool(s.alphas_cumprod[-1] < 1e-9) is zero_terminal
    assert s.config.prediction_type == ("v_prediction" if m.v_pred else "epsilon")


# ---- previews ---------------------------------------------------------------------------------


@pytest.mark.parametrize("clip_skip", [None, 2])
def test_sdxl_preview_passes_training_clip_skip_to_diffusers(clip_skip, monkeypatch):
    from rengu_flow.utils import preview

    pipe = MagicMock()
    pipe.return_value.images = [object()]
    model = SimpleNamespace(load_diffusion_model=lambda: None, _pipeline=pipe, clip_skip=clip_skip)
    monkeypatch.setattr(preview, "_log_preview_image", lambda **kw: None)
    preview._run_sdxl_previews(model, {"width": 64, "height": 64}, [("p", "a cat")], sink=None, step=1)
    assert pipe.call_args.kwargs["clip_skip"] == clip_skip


# ---- text-cache key ----------------------------------------------------------------------------


def test_sdxl_text_cache_key_covers_clip_skip_encoding_version_and_checkpoint(tmp_path):
    cap = get_capability("sdxl")
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"a" * 10)
    base = {"type": "sdxl", "checkpoint_path": str(ckpt)}
    ident = cap.text_cache_identity(base)
    assert ident["text_cache_version"] == "1"  # pad-to-77 encoding: pre-padding caches get re-encoded
    assert "clip_skip" not in ident
    assert cap.text_cache_identity({**base, "clip_skip": 2}) != ident
    assert cap.text_cache_identity({**base, "clip_skip": 2})["clip_skip"] == "2"
    # replacing the file (different size) changes the key; moving it (same size+mtime) does not
    ckpt.write_bytes(b"a" * 11)
    assert cap.text_cache_identity(base) != ident
    moved = tmp_path / "moved.safetensors"
    moved.write_bytes(ckpt.read_bytes())
    import os

    st = ckpt.stat()
    os.utime(moved, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert cap.text_cache_identity({**base, "checkpoint_path": str(moved)}) == cap.text_cache_identity(base)


def test_sdxl_cache_fingerprint_args_include_identity(tmp_path):
    from rengu_flow.data.dataset import text_cache_fingerprint_args

    a = text_cache_fingerprint_args({"model": {"type": "sdxl", "checkpoint_path": "x"}})
    b = text_cache_fingerprint_args({"model": {"type": "sdxl", "checkpoint_path": "x", "clip_skip": 2}})
    assert a and b and a != b


def test_cosmos_text_cache_key_follows_the_encoder_file(tmp_path):
    cap = get_capability("cosmos_predict2")
    llm = tmp_path / "qwen.safetensors"
    llm.write_bytes(b"1" * 8)
    t5 = tmp_path / "t5.safetensors"
    t5.write_bytes(b"2" * 20)
    only_llm = cap.text_cache_identity({"llm_path": str(llm)})
    assert only_llm["text_encoder_file"].startswith("llm_path:")
    # t5_path wins in load_text_stack, so it decides the key
    both = cap.text_cache_identity({"llm_path": str(llm), "t5_path": str(t5)})
    assert both["text_encoder_file"].startswith("t5_path:") and both != only_llm
    llm.write_bytes(b"1" * 9)
    assert cap.text_cache_identity({"llm_path": str(llm)}) != only_llm
    # a folder-style llm_path is fingerprinted by its files
    d = tmp_path / "qwen_dir"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    id_dir = cap.text_cache_identity({"llm_path": str(d)})
    (d / "model.safetensors").write_bytes(b"xy")
    assert cap.text_cache_identity({"llm_path": str(d)}) != id_dir


def test_models_without_text_cache_declarations_keep_their_key():
    assert get_capability("krea2").text_cache_identity({}) == {}
    assert get_capability("sdxl").text_cache_identity({}) == {"text_cache_version": "1"}

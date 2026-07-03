"""Host-side preflight: catch what WILL fail before any model loads."""

from __future__ import annotations

import os

import pytest

from rengu_flow.config.preflight import collect_preflight_issues


@pytest.fixture(autouse=True)
def _enable_preflight(monkeypatch):
    monkeypatch.setenv("RENGU_PREFLIGHT", "1")


def _config(tmp_path, **overrides):
    """A config whose referenced files all exist (zero issues), to mutate per test."""
    files = {}
    for name in ("dit.safetensors", "vae.safetensors", "te.safetensors"):
        f = tmp_path / name
        f.write_bytes(b"x")
        files[name] = str(f)
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    ds = tmp_path / "data.toml"
    ds.write_text(f'resolutions = [512]\n[[directory]]\npath = "{img_dir}"\n')
    cfg = {
        "output_dir": str(tmp_path / "output"),
        "dataset": str(ds),
        "model": {
            "type": "krea2",
            "dtype": "bfloat16",
            "transformer_path": files["dit.safetensors"],
            "vae_path": files["vae.safetensors"],
            "text_encoder_path": files["te.safetensors"],
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }
    cfg.update(overrides)
    return cfg


def test_clean_config_has_no_issues(tmp_path):
    assert collect_preflight_issues(_config(tmp_path)) == []


def test_missing_model_component_paths(tmp_path):
    cfg = _config(tmp_path)
    cfg["model"]["vae_path"] = str(tmp_path / "nope.safetensors")
    issues = collect_preflight_issues(cfg)
    assert any("model.vae_path does not exist" in i for i in issues)


def test_missing_dataset_toml_and_directory(tmp_path):
    cfg = _config(tmp_path, dataset=str(tmp_path / "missing.toml"))
    assert any("dataset TOML does not exist" in i for i in collect_preflight_issues(cfg))

    ds = tmp_path / "bad_dir.toml"
    ds.write_text(f'resolutions = [512]\n[[directory]]\npath = "{tmp_path / "ghost"}"\n')
    cfg = _config(tmp_path, dataset=str(ds))
    assert any("is not a directory" in i for i in collect_preflight_issues(cfg))


def test_adapter_init_and_resume_paths(tmp_path):
    cfg = _config(tmp_path)
    cfg["adapter"] = {"type": "lokr", "rank": 4, "init_from_existing": str(tmp_path / "ghost")}
    cfg["resume_from_checkpoint"] = str(tmp_path / "ckpt-missing")
    issues = collect_preflight_issues(cfg)
    assert any("adapter.init_from_existing does not exist" in i for i in issues)
    assert any("resume_from_checkpoint does not exist" in i for i in issues)
    # boolean resume ("--resume_from_checkpoint" bare / latest) is not a path
    cfg["resume_from_checkpoint"] = True
    cfg["adapter"].pop("init_from_existing")
    assert not any("resume_from_checkpoint" in i for i in collect_preflight_issues(cfg))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
def test_unwritable_output_dir(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        cfg = _config(tmp_path, output_dir=str(locked / "runs"))
        assert any("output_dir is not writable" in i for i in collect_preflight_issues(cfg))
    finally:
        locked.chmod(0o700)


def test_target_pattern_typo_caught_torch_free(tmp_path):
    """'text_fussion.*' fails preflight with the model's real module roots — no model load."""
    cfg = _config(tmp_path)
    cfg["adapter"] = {"type": "lokr", "rank": 4, "target_include": ["text_fussion.*"]}
    issues = collect_preflight_issues(cfg)
    assert any("text_fussion" in i and "Roots:" in i for i in issues)
    # glob first segments and valid roots pass
    cfg["adapter"]["target_include"] = ["*attn*", "text_fusion.*"]
    assert not any("matches nothing" in i for i in collect_preflight_issues(cfg))


def test_conflicting_loss_switches(tmp_path):
    cfg = _config(tmp_path, huber_delta=0.1, pseudo_huber_c=0.03)
    assert any("Only one loss switch" in i for i in collect_preflight_issues(cfg))


def test_gradient_release_engine_combo(tmp_path, monkeypatch):
    monkeypatch.delenv("RENGU_ENGINE", raising=False)
    cfg = _config(tmp_path, engine="accelerate")
    cfg["optimizer"]["gradient_release"] = True
    assert any("gradient_release requires engine='deepspeed'" in i for i in collect_preflight_issues(cfg))
    cfg["engine"] = "deepspeed"
    assert not any("gradient_release" in i for i in collect_preflight_issues(cfg))


def test_engine_value_and_scheduler_name(tmp_path):
    cfg = _config(tmp_path, engine="acelerate")  # typo
    assert any("engine 'acelerate'" in i for i in collect_preflight_issues(cfg))
    cfg = _config(tmp_path, lr_scheduler="cosinus")
    assert any("lr_scheduler 'cosinus' is not registered" in i for i in collect_preflight_issues(cfg))
    cfg = _config(tmp_path, lr_scheduler="cosine")
    assert not any("lr_scheduler" in i for i in collect_preflight_issues(cfg))


def test_async_export_with_pipeline_stages(tmp_path):
    cfg = _config(tmp_path, async_model_export=True, pipeline_stages=2)
    assert any("async_model_export" in i for i in collect_preflight_issues(cfg))


def test_preview_enabled_without_prompts(tmp_path):
    cfg = _config(tmp_path, preview={"enabled": True, "prompts": []})
    assert any("preview.prompts is empty" in i for i in collect_preflight_issues(cfg))


def test_module_roots_match_real_krea2_dit():
    """The torch-free roots in the capability must track the real DiT's top-level modules."""
    import torch  # noqa: F401  (model import needs torch)

    from rengu_flow.registry.model_capabilities import get_capability
    from tests.test_adapter_layer_groups import _tiny_krea2

    roots = set(get_capability("krea2").adapter_module_roots)
    model = _tiny_krea2()
    real_roots = {name for name, _ in model.named_children()}
    with_params = {
        n for n in real_roots
        if any(True for _ in getattr(model, n).parameters())
    }
    missing = with_params - roots
    assert not missing, f"capability adapter_module_roots is missing: {sorted(missing)}"

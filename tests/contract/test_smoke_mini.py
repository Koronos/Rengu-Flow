"""Contract E: the smoke tooling follows the registered models.

Every model with a test kit runs in ``scripts/smoke_mini.py`` (in process, CPU, one step); every
registered model has a real-weights smoke fixture ``tests/fixtures/smoke/train_<type>.toml`` or its
kit declares ``SMOKE_FIXTURE_EXEMPT``; the docs name the generic smoke commands (no per-model
lists that must be kept in sync); the generic export check (``rengu_flow.networks.export_check``)
rejects what it claims to."""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path

import pytest

from contract_helpers import import_or_skip
from rengu_flow.registry.builtin_models import builtin_specs

pytestmark = pytest.mark.no_ui_db

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "smoke"
SPECS = [spec for spec in builtin_specs() if spec.pipeline]
KIT_SPECS = [spec for spec in SPECS if spec.testkit]


def _smoke_mini():
    spec = importlib.util.spec_from_file_location("smoke_mini", REPO_ROOT / "scripts" / "smoke_mini.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("spec", KIT_SPECS, ids=[s.type_id for s in KIT_SPECS])
def test_every_testkit_model_runs_in_smoke_mini(spec, capsys):
    import_or_skip(spec.pipeline_module)
    adapter = "lora" if "lora" in spec.adapters else "full"
    assert _smoke_mini().main(["--model", spec.type_id, "--adapter", adapter, "--steps", "1"]) == 0
    out = capsys.readouterr().out
    assert f"Smoke mini {spec.type_id} ({adapter}) OK" in out, out
    assert "[mini] export OK" in out, out


@pytest.mark.parametrize("spec", SPECS, ids=[s.type_id for s in SPECS])
def test_model_has_a_smoke_fixture_or_declares_the_exemption(spec):
    fixture = FIXTURES / f"train_{spec.type_id}.toml"
    exempt = getattr(importlib.import_module(spec.testkit), "SMOKE_FIXTURE_EXEMPT", None) if spec.testkit else None
    if exempt is not None:
        assert isinstance(exempt, str) and exempt, "SMOKE_FIXTURE_EXEMPT is the reason (a non-empty string)"
        assert not fixture.exists(), f"{fixture.name} exists: drop SMOKE_FIXTURE_EXEMPT from {spec.testkit}"
    else:
        # cosmos_predict2 is also run as the legacy short name `cosmos` (run_model_smoke.sh cosmos).
        assert fixture.is_file(), f"add {fixture.relative_to(REPO_ROOT)} (no [model] paths) or declare SMOKE_FIXTURE_EXEMPT in the kit"
        import toml

        assert str(toml.load(fixture)["model"]["type"]).lower() == spec.type_id


@pytest.mark.parametrize("doc", ["AGENTS.md", "docs/developer/smoke-tests.md"])
def test_docs_name_the_generic_smoke_commands(doc):
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    assert "scripts/smoke_mini.py --model <model>" in text
    assert re.search(r"run_model_smoke\.sh <model>", text)
    # No per-model alternation (`run_model_smoke.sh a|b|c`) that has to be kept in sync.
    assert not re.search(r"run_model_smoke\.sh [a-z0-9_]+(\\?\|)[a-z0-9_]+", text), "list models generically"
    for gone in ("smoke_krea2_mini", "smoke_nanosaur2_mini"):
        assert gone not in text


def test_run_model_smoke_has_no_per_model_cases():
    script = (REPO_ROOT / "scripts" / "run_model_smoke.sh").read_text(encoding="utf-8")
    for spec in SPECS:
        assert not re.search(rf"^\s*{spec.type_id}\)", script, re.M), f"{spec.type_id} is a default fixture case"
    assert "train_${MODEL}" in script or "train_${candidate}" in script
    assert not re.search(r"^(?!lycoris)\w+_export_check\(\)", script, re.M), "export checks are generic (export_check.py)"


# ---- the generic export check ---------------------------------------------------------------------


def _lora(prefix: str, names=("a", "b")) -> dict:
    import torch

    state = {}
    for n in names:
        state[f"{prefix}{n}.lora_A.weight"] = torch.ones(2, 3)
        state[f"{prefix}{n}.lora_B.weight"] = torch.ones(3, 2)
        state[f"{prefix}{n}.alpha"] = torch.tensor(4.0)
    return state


def _kohya_lora(names=("lora_unet_a", "lora_te1_text_model_b")) -> dict:
    import torch

    state = {}
    for n in names:
        state[f"{n}.lora_down.weight"] = torch.ones(2, 3)
        state[f"{n}.lora_up.weight"] = torch.ones(3, 2)
        state[f"{n}.alpha"] = torch.tensor(2.0)
    return state


def test_export_check_accepts_a_prefixed_lora_and_rejects_each_defect():
    import torch

    from rengu_flow.networks.export_check import adapter_state_problems as problems
    from rengu_flow.registry.model_spec import AdapterExport

    dotted = AdapterExport(roots=("p.",))
    assert problems(_lora("p."), dotted, "lora") == []
    assert problems(_lora("whatever."), AdapterExport(), "lora") == []  # no declared roots: not checked
    assert problems({}, dotted, "lora")
    assert any("roots" in p for p in problems(_lora("other."), dotted, "lora"))
    unpaired = _lora("p.")
    del unpaired["p.a.lora_B.weight"]
    assert any("unpaired" in p for p in problems(unpaired, dotted, "lora"))
    renamed = {k.replace("lora_A.weight", "lora_A.default.weight"): v for k, v in _lora("p.").items()}
    assert any("LoRA key names" in p for p in problems(renamed, dotted, "lora"))
    nan = _lora("p.")
    nan["p.a.lora_A.weight"] = torch.full((2, 3), float("nan"))
    assert any("non-finite" in p for p in problems(nan, dotted, "lora"))
    lokr = {"p.a.lokr_w1": torch.ones(2, 2), "p.a.lokr_w2": torch.ones(2, 2), "p.a.alpha": torch.tensor(1.0)}
    assert problems(lokr, dotted, "lokr") == []


def test_export_check_kohya_style_rejects_each_defect():
    """Kohya layout (SDXL LoRA / LyCORIS): flat modules under the roots, each with exactly its
    family's weights; the first root (the denoiser's) must be present."""
    import torch

    from rengu_flow.networks.export_check import adapter_state_problems as problems
    from rengu_flow.registry.model_spec import AdapterExport

    kohya = AdapterExport(file="lora.safetensors", style="kohya", roots=("lora_unet_", "lora_te1_text_model_"))
    assert problems(_kohya_lora(), kohya, "lora") == []
    assert any("roots" in p for p in problems({**_kohya_lora(), **_kohya_lora(("lora_te3_x",))}, kohya, "lora"))
    assert any("'lora_unet_' keys" in p for p in problems(_kohya_lora(("lora_te1_text_model_b",)), kohya, "lora"))
    no_up = _kohya_lora()
    del no_up["lora_unet_a.lora_up.weight"]
    assert any("missing" in p for p in problems(no_up, kohya, "lora"))
    peft_names = {k.replace("lora_down", "lora_A"): v for k, v in _kohya_lora().items()}
    assert any("unexpected weights" in p for p in problems(peft_names, kohya, "lora"))
    loha = {f"lora_unet_a.{w}": torch.ones(2, 2) for w in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "alpha")}
    assert problems(loha, kohya, "lycoris_loha") == []
    assert any("missing" in p for p in problems(loha, kohya, "lycoris_locon"))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.type_id)
def test_export_check_runs_on_a_run_output_dir(spec, tmp_path):
    """``check_run`` finds the newest adapter export (the file the spec's layout names) under an
    output dir and reads the model type and adapter kind from the training config."""
    from safetensors.torch import save_file

    from rengu_flow.networks.export_check import check_run

    layout = spec.adapter_export("lora")
    config = tmp_path / "train.toml"
    config.write_text(f'[model]\ntype = "{spec.type_id}"\n[adapter]\ntype = "lora"\n', encoding="utf-8")
    out = tmp_path / "output" / "run" / "epoch1"
    out.mkdir(parents=True)
    good = _kohya_lora((layout.roots[0] + "a",)) if layout.style == "kohya" else _lora(spec.export_prefix or "")
    save_file(good, str(out / layout.file))
    assert spec.type_id in check_run(config, tmp_path / "output")
    save_file({"wrong." + k: v for k, v in good.items()}, str(out / layout.file))
    with pytest.raises(AssertionError, match="roots|unexpected|missing|keys"):
        check_run(config, tmp_path / "output")
    (out / layout.file).rename(out / "renamed.safetensors")
    with pytest.raises(AssertionError, match=re.escape(f"no {layout.file}")):
        check_run(config, tmp_path / "output")

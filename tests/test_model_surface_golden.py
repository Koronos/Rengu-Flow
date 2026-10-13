"""Golden harness for the per-model integration surface.

``scripts/dump_model_surface.py`` serialises everything a model type touches outside its own
package (capability descriptor, UI schema/visibility, preview form, ``set_config_defaults``,
validation rules, env-var path map, installer profiles, preview dispatch) to
``tests/golden/model_surface/<type>.json`` plus ``_global.json``. This test recomputes the dump
and compares it section by section, so a refactor meant to preserve behavior fails with a readable
diff on whatever moved. Legacy TOML shapes live in ``tests/golden/legacy_configs/``.

Intentional change? Regenerate and review the git diff of ``tests/golden``::

    RENGU_UPDATE_GOLDEN=1 uv run --no-sync pytest tests/test_model_surface_golden.py -n0
    # or: uv run --no-sync python scripts/dump_model_surface.py
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from rengu_flow.registry.model_capabilities import get_canonical_model_types

pytestmark = pytest.mark.no_ui_db

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
SURFACE_DIR = GOLDEN_DIR / "model_surface"
LEGACY_DIR = GOLDEN_DIR / "legacy_configs"
LEGACY_EXPECTED = LEGACY_DIR / "expected.json"
UPDATE = os.environ.get("RENGU_UPDATE_GOLDEN") == "1"
REGEN_HINT = (
    "If the change is intentional, regenerate with RENGU_UPDATE_GOLDEN=1 "
    "(uv run --no-sync pytest tests/test_model_surface_golden.py -n0) or "
    "`uv run --no-sync python scripts/dump_model_surface.py`, then review the diff of tests/golden."
)
MAX_DIFF_LINES = 60


def _load_dumper():
    spec = importlib.util.spec_from_file_location("dump_model_surface", REPO_ROOT / "scripts" / "dump_model_surface.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("dump_model_surface", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dumper():
    return _load_dumper()


@pytest.fixture(scope="module")
def current(dumper):
    """Live dump of every canonical type + ``_global`` (computed once per module/worker)."""
    return dumper.dump_all()


@pytest.fixture(scope="module")
def current_legacy(dumper):
    return dumper.dump_legacy(LEGACY_DIR)


def _pretty(obj) -> list[str]:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).splitlines()


def _read_golden(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_diff(exp, act, path: str = "") -> list[str]:
    """``path: golden -> current`` lines for every leaf that differs (lists compared whole)."""
    if isinstance(exp, dict) and isinstance(act, dict):
        out: list[str] = []
        for key in sorted(set(exp) | set(act)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in exp:
                out.append(f"+ {sub}: {json.dumps(act[key], ensure_ascii=False)[:160]}")
            elif key not in act:
                out.append(f"- {sub}: {json.dumps(exp[key], ensure_ascii=False)[:160]}")
            else:
                out.extend(_deep_diff(exp[key], act[key], sub))
        return out
    if exp == act:
        return []
    if isinstance(exp, list) and isinstance(act, list) and all(isinstance(i, (str, int)) for i in exp + act):
        added = [i for i in act if i not in exp]
        removed = [i for i in exp if i not in act]
        note = "reordered" if not added and not removed else f"added {added} removed {removed}"
        return [f"~ {path}: list {note}"]
    return [f"~ {path}: {json.dumps(exp, ensure_ascii=False)[:160]} -> {json.dumps(act, ensure_ascii=False)[:160]}"]


def _section_diffs(expected: dict, actual: dict) -> list[str]:
    """One readable unified diff per top-level section that differs."""
    out: list[str] = []
    for section in sorted(set(expected) | set(actual)):
        exp = expected.get(section, "<section missing in golden>")
        act = actual.get(section, "<section missing in current dump>")
        if exp == act:
            continue
        diff = list(
            difflib.unified_diff(_pretty(exp), _pretty(act), "golden", "current", lineterm="", n=2)
        )
        shown = diff[:MAX_DIFF_LINES]
        more = len(diff) - len(shown)
        where = _deep_diff(exp, act)
        where_shown = where[:25] + ([f"... ({len(where) - 25} more)"] if len(where) > 25 else [])
        out.append(
            f"--- section [{section}] differs ---\nwhere (- golden only, + current only, ~ changed):\n  "
            + "\n  ".join(where_shown)
            + "\nunified diff:\n"
            + "\n".join(shown)
            + (f"\n... ({more} more diff lines)" if more > 0 else "")
        )
    return out


def _write(path: Path, data) -> None:
    from dump_model_surface import dumps  # loaded by the `dumper` fixture

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(data), encoding="utf-8", newline="\n")


def _assert_matches(path: Path, actual, name: str) -> None:
    if UPDATE:
        _write(path, actual)
        pytest.skip(f"golden regenerated: {path.relative_to(REPO_ROOT)}")
    if not path.is_file():
        pytest.fail(
            f"No golden for {name!r} ({path.relative_to(REPO_ROOT)}). Registered model types need one. "
            + REGEN_HINT,
            pytrace=False,
        )
    expected = _read_golden(path)
    actual = json.loads(json.dumps(actual, sort_keys=True))
    if expected == actual:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        detail = "\n\n".join(_section_diffs(expected, actual))
    else:
        detail = "\n".join(list(difflib.unified_diff(_pretty(expected), _pretty(actual), "golden", "current", lineterm=""))[:MAX_DIFF_LINES])
    pytest.fail(f"Model surface for {name!r} drifted from its golden.\n\n{detail}\n\n{REGEN_HINT}", pytrace=False)


@pytest.mark.parametrize("model_type", [*get_canonical_model_types(), "_global"])
def test_model_surface_matches_golden(current, model_type):
    if model_type not in current:  # registry changed between collection and dump (should not happen)
        pytest.fail(f"{model_type!r} missing from the live dump", pytrace=False)
    _assert_matches(SURFACE_DIR / f"{model_type}.json", current[model_type], model_type)


def test_every_registered_type_has_a_golden():
    if UPDATE:
        pytest.skip("regenerating")
    missing = [t for t in get_canonical_model_types() if not (SURFACE_DIR / f"{t}.json").is_file()]
    assert not missing, f"registered model types without a golden: {missing}. {REGEN_HINT}"


def test_no_orphan_goldens():
    """A golden for a type that is no longer registered means the type vanished (or was renamed)."""
    if UPDATE:
        pytest.skip("regenerating")
    on_disk = {p.stem for p in SURFACE_DIR.glob("*.json")} - {"_global"}
    orphans = sorted(on_disk - set(get_canonical_model_types()))
    assert not orphans, f"goldens for unregistered model types: {orphans}. {REGEN_HINT}"


def test_legacy_configs_match_golden(current_legacy):
    """Validation / defaults / path / installer outcomes for real legacy config shapes."""
    if UPDATE:
        _write(LEGACY_EXPECTED, current_legacy)
        pytest.skip("legacy golden regenerated")
    assert LEGACY_EXPECTED.is_file(), f"missing {LEGACY_EXPECTED}. {REGEN_HINT}"
    expected = _read_golden(LEGACY_EXPECTED)
    actual = json.loads(json.dumps(current_legacy, sort_keys=True))
    if expected == actual:
        return
    parts = []
    for name in sorted(set(expected) | set(actual)):
        if expected.get(name) == actual.get(name):
            continue
        if name not in expected or name not in actual:
            parts.append(f"--- {name}: " + ("new config without expected entry" if name not in expected else "expected entry has no config"))
            continue
        parts.append(f"=== legacy config {name} ===\n" + "\n\n".join(_section_diffs(expected[name], actual[name])))
    pytest.fail("Legacy config outcomes drifted.\n\n" + "\n\n".join(parts) + "\n\n" + REGEN_HINT, pytrace=False)


def test_legacy_corpus_covers_the_known_shapes():
    """The corpus must keep exercising the shapes the refactor must not break."""
    names = {p.stem for p in LEGACY_DIR.glob("*.toml")}
    for required in (
        "cosmos_alias_anima_rejected",
        "cosmos_t5_path_instead_of_llm",
        "krea2_checkpoint_path_only",
        "qwen_image21_diffusers_path_only",
        "sdxl_adapter_dim_alias",
        "sdxl_ac_selective",
    ):
        assert required in names, f"legacy corpus lost {required}.toml"
    expected = _read_golden(LEGACY_EXPECTED)
    anima = expected["cosmos_alias_anima_rejected.toml"]["validate_config_model_rules"]
    assert "Unknown model type 'anima'" in anima.get("error", ""), "raw 'anima' must stay rejected"

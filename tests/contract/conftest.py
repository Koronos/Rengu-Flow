"""Contract suite over every registered model with a test kit (``spec.testkit``).

Each check is parametrized over the canonical model types whose spec names a kit
(``rengu_flow/model/<m>/testkit.py``); a model whose extras are not installed is skipped. A check
that does not apply to a model is declared by its kit (``CAPABILITIES``, ``cases()``), never by
the model type. CPU only: CUDA is hidden for the whole suite.

Numeric goldens (``tests/golden/model_numeric/<type>.json``) pin tiny-model outputs so a
refactor of shared pipeline code (docs/spec/model-spec-refactor.md, phase 4) cannot silently
change numerics. Regenerate with ``RENGU_UPDATE_GOLDEN=1`` and ``-n 0`` (one writer per file).
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from contract_helpers import close, import_or_skip

# Contract A reuses the knob-coverage helpers of tests/test_ui_knob_coverage.py.
_TESTS_DIR = str(Path(__file__).resolve().parents[1])
if _TESTS_DIR not in sys.path:
    sys.path.append(_TESTS_DIR)

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "model_numeric"
UPDATE_GOLDEN = os.environ.get("RENGU_UPDATE_GOLDEN") == "1"


def _kit_specs():
    from rengu_flow.registry.builtin_models import builtin_specs

    return [spec for spec in builtin_specs() if spec.testkit]


KIT_SPECS = _kit_specs()


class Kit:
    """A model's test kit plus its spec and a per-session folder of tiny weights."""

    def __init__(self, spec, module, root: Path):
        self.spec = spec
        self.module = module
        self.root = root
        self.type_id = spec.type_id
        self.capabilities = module.CAPABILITIES
        self.cases = module.cases()

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, capability: str) -> None:
        if not self.has(capability):
            reason = getattr(self.module, "NOT_APPLICABLE", {}).get(capability, "no reason declared")
            pytest.skip(f"{self.type_id}: {capability!r} does not apply: {reason}")

    def config(self, **over) -> dict:
        return copy.deepcopy(self.module.tiny_config(self.root, **over))

    def pipeline(self, **over):
        return self.module.tiny_pipeline(self.config(**over))

    def denoiser(self, pipeline):
        """The trained network: ``pipeline.transformer`` (DiTs) unless the kit names another."""
        hook = getattr(self.module, "denoiser", None)
        return hook(pipeline) if hook else pipeline.transformer

    def adapter_networks(self, pipeline) -> dict:
        """``{original_name prefix: module}`` the adapters attach to (DiTs: the transformer)."""
        hook = getattr(self.module, "adapter_networks", None)
        return hook(pipeline) if hook else {"": pipeline.transformer}

    def adapter_scope(self, pipeline, adapter: str) -> list[str]:
        """Module names an ``adapter`` export must cover, without ``spec.export_prefix`` (DiTs:
        the linears of ``adapter_target_modules`` minus the kit's ``ADAPTER_SCOPE_EXCLUDES``)."""
        hook = getattr(self.module, "adapter_scope", None)
        if hook:
            return sorted(hook(pipeline, adapter))
        from rengu_flow.networks import adapter_dit

        excluded = tuple(getattr(self.module, "ADAPTER_SCOPE_EXCLUDES", {}).get(adapter, ()))
        names = adapter_dit._collect_target_linears(pipeline.transformer, tuple(pipeline.adapter_target_modules))
        return sorted(n for n in names if not n.startswith(excluded))

    def reload_export(self, save_dir):
        """A pipeline loaded from ``save_model``'s output in ``save_dir``."""
        hook = getattr(self.module, "reload_export", None)
        if hook:
            return hook(self.config(), save_dir)
        return self.pipeline(transformer_path=str(self.module.export_transformer_path(save_dir)))

    def case(self, name: str):
        return next(c for c in self.cases if c.name == name)

    @contextlib.contextmanager
    def known_gap(self, key: str):
        """Assertions in this block may fail for a gap the kit declares (``KNOWN_GAPS[key]``):
        then the test xfails with that reason; once it passes, the stale declaration fails."""
        reason = getattr(self.module, "KNOWN_GAPS", {}).get(key)
        if reason is None:
            yield
            return
        try:
            yield
        except AssertionError:
            pytest.xfail(f"{self.type_id} known gap {key!r}: {reason}")
        pytest.fail(f"{self.type_id}: known gap {key!r} no longer reproduces; remove it from KNOWN_GAPS")


_ROOTS: dict[str, Path] = {}


def _load_kit(spec, tmp_path_factory) -> Kit:
    import_or_skip(spec.pipeline_module)
    module = importlib.import_module(spec.testkit)
    if spec.type_id not in _ROOTS:
        _ROOTS[spec.type_id] = tmp_path_factory.mktemp(f"tiny_{spec.type_id}")
    return Kit(spec, module, _ROOTS[spec.type_id])


def _case_params():
    params = []
    for spec in KIT_SPECS:
        names = [c.name for c in importlib.import_module(spec.testkit).cases()]  # kits import no torch
        params += [pytest.param((spec, name), id=f"{spec.type_id}-{name}") for name in names]
    return params


@pytest.fixture(params=KIT_SPECS, ids=[spec.type_id for spec in KIT_SPECS])
def kit(request, tmp_path_factory) -> Kit:
    return _load_kit(request.param, tmp_path_factory)


@pytest.fixture(params=_case_params())
def kit_case(request, tmp_path_factory):
    """``(kit, case)`` for every case of every kit."""
    spec, name = request.param
    k = _load_kit(spec, tmp_path_factory)
    return k, k.case(name)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """The contract suite is CPU-only (no GPU needed, deterministic numerics). Spec-only checks
    (contract A) never import torch; they get no patch."""
    torch = sys.modules.get("torch")
    if torch is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def golden():
    """``golden(type_id, key, value, rel=1e-4, abs_=1e-5)``: compare ``value`` (JSON-able) with
    ``tests/golden/model_numeric/<type_id>.json[key]``; ``RENGU_UPDATE_GOLDEN=1`` rewrites it."""

    def check(type_id: str, key: str, value, rel: float = 1e-4, abs_: float = 1e-5) -> None:
        path = GOLDEN_DIR / f"{type_id}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if UPDATE_GOLDEN:
            if os.environ.get("PYTEST_XDIST_WORKER"):
                pytest.fail("regenerate the numeric goldens with -n 0 (one writer per golden file)")
            data[key] = value
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(sorted(data.items())), indent=1) + "\n", encoding="utf-8")
            return
        assert key in data, f"no golden {key!r} in {path.name}: run with RENGU_UPDATE_GOLDEN=1 -n 0"
        assert close(value, data[key], rel, abs_), (
            f"{type_id} {key} drifted from the golden (RENGU_UPDATE_GOLDEN=1 -n 0 to accept):\n"
            f"  got      {value}\n  expected {data[key]}"
        )

    return check

"""Validate every TOML under examples/ (schema + UI probe).

Model checkpoint paths may be placeholders (e.g. path/to/sdxl.safetensors);
validation does not require files on disk. Dataset examples may use placeholder
directory paths — only structural / augmentation schema is checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import toml

from rengu_flow.config import load_config, load_dataset_config, set_config_defaults, validate_config
from rengu_flow.config.validation import collect_validation_errors
from rengu_flow_ui.run_staging import validate_toml_text
from rengu_flow_ui.datasets_store import validate_dataset_text

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _REPO_ROOT / "examples"

def _is_training_example(path: Path) -> bool:
    """A training config has a ``[model]`` table; a dataset TOML does not."""
    return isinstance(toml.load(path).get("model"), dict)


# Derived from the files (no per-model list): every example is one or the other.
TRAINING_EXAMPLES = sorted(p.name for p in _EXAMPLES.glob("*.toml") if _is_training_example(p))
DATASET_EXAMPLES = sorted(p.name for p in _EXAMPLES.glob("*.toml") if not _is_training_example(p))


def test_examples_cover_every_registered_model() -> None:
    """Each registered model has a training example and each example's dataset ships too."""
    from rengu_flow.registry.model_capabilities import model_capability_registry

    types = {toml.load(_EXAMPLES / name)["model"]["type"] for name in TRAINING_EXAMPLES}
    assert {t for t, cap in model_capability_registry.items() if cap.pipeline} <= types
    assert DATASET_EXAMPLES


@pytest.mark.parametrize("config_name", TRAINING_EXAMPLES)
def test_training_example_toml_parses(config_name: str) -> None:
    path = _EXAMPLES / config_name
    assert path.is_file(), f"missing {path}"
    toml.load(path.open(encoding="utf-8"))


@pytest.mark.parametrize("config_name", TRAINING_EXAMPLES)
def test_training_example_schema_and_ui_validate(config_name: str) -> None:
    path = _EXAMPLES / config_name
    text = path.read_text(encoding="utf-8")
    config = load_config(path)
    issues = collect_validation_errors(config)
    assert not issues, issues
    set_config_defaults(config)
    validate_config(config)
    ui = validate_toml_text(text)
    assert ui["ok"], ui.get("errors") or ui.get("error")
    dataset = config.get("dataset")
    if isinstance(dataset, str) and (_REPO_ROOT / dataset).is_file():
        config["dataset"] = str(_REPO_ROOT / dataset)
        assert load_dataset_config(config) is not None


@pytest.mark.parametrize("config_name", TRAINING_EXAMPLES)
def test_training_example_main_validate_only(config_name: str) -> None:
    path = _EXAMPLES / config_name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rengu_flow.main",
            "--config",
            str(path),
            "--validate-only",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


def _registered_model_types() -> list[str]:
    from rengu_flow.registry.model_capabilities import model_capability_registry

    return sorted(t for t, cap in model_capability_registry.items() if cap.pipeline)


@pytest.mark.parametrize("model_type", _registered_model_types())
def test_adapter_example_of_every_registered_model_passes_main_validate_only(model_type: str) -> None:
    """For each registered model (from the registry, not a list here), an adapter example exists
    and ``python -m rengu_flow.main --validate-only`` on it exits 0."""
    adapter_examples = [
        name
        for name in TRAINING_EXAMPLES
        if str(toml.load(_EXAMPLES / name)["model"].get("type", "")).lower() == model_type
        and isinstance(toml.load(_EXAMPLES / name).get("adapter"), dict)
    ]
    assert adapter_examples, f"no adapter example (a [adapter] table) for {model_type} under examples/"
    result = subprocess.run(
        [sys.executable, "-m", "rengu_flow.main", "--config", str(_EXAMPLES / adapter_examples[0]), "--validate-only"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")


@pytest.mark.parametrize("dataset_name", sorted(DATASET_EXAMPLES))
def test_dataset_example_validates(dataset_name: str) -> None:
    path = _EXAMPLES / dataset_name
    text = path.read_text(encoding="utf-8")
    toml.loads(text)
    result = validate_dataset_text(text)
    assert result["ok"], result.get("error") or result.get("errors")


def test_main_validate_only_does_not_load_the_dataset(tmp_path) -> None:
    """With --validate-only, main loads and validates the config and exits 0 without reading the
    dataset (here a path that does not exist)."""
    config_path = tmp_path / "validate_only.toml"
    config_path.write_text(
        "\n".join(
            [
                'dataset = "nonexistent_dataset.toml"',
                "[model]",
                'type = "sdxl"',
                'dtype = "bfloat16"',
                'checkpoint_path = "path/to/sdxl.safetensors"',
                "[optimizer]",
                'type = "adamw"',
                "lr = 1.0e-4",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path), "--validate-only"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

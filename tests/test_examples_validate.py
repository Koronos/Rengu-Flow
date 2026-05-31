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

from rengu_flow.config import load_config, set_config_defaults, validate_config
from rengu_flow.config.validation import collect_validation_errors
from rengu_flow_ui.run_staging import validate_toml_text
from rengu_flow_ui.datasets_store import validate_dataset_text

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES = _REPO_ROOT / "examples"

# Dataset-only TOMLs (no [model] / training sections).
DATASET_EXAMPLES = frozenset(
    {
        "minimal_dataset.toml",
        "minimal_cosmos_predict2_dataset.toml",
        "dataset_augmentation_research.toml",
    }
)

TRAINING_EXAMPLES = sorted(
    p.name for p in _EXAMPLES.glob("*.toml") if p.name not in DATASET_EXAMPLES
)


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


@pytest.mark.parametrize("dataset_name", sorted(DATASET_EXAMPLES))
def test_dataset_example_validates(dataset_name: str) -> None:
    path = _EXAMPLES / dataset_name
    text = path.read_text(encoding="utf-8")
    toml.loads(text)
    result = validate_dataset_text(text)
    assert result["ok"], result.get("error") or result.get("errors")

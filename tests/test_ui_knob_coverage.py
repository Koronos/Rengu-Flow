"""Regression guard: every trainer-consumed config key must be exposed in the UI.

Extracts config keys the trainer actually *reads* (via a receiver-name regex over
``rengu_flow/**/*.py``, skipping ``vendor/`` and ``tests/``) and checks each one is either:

  * a top-level field path in ``rengu_flow_ui.config_schema.get_sections()``,
  * a ``model.*`` field path in some capability's ``model_fields``
    (``rengu_flow.registry.model_capabilities``), or
  * explicitly allowlisted below as an intentional non-UI key.

This is a heuristic (regex, not a real static analyzer): it catches obvious "we added a
config.get(...) call but forgot the UI field" regressions, not every possible access pattern.
False negatives (a consumed key the regex misses) are an accepted limitation; false positives
(a key genuinely covered elsewhere) go in ALLOWLIST with a one-line reason, not by weakening
the regex.
"""

from __future__ import annotations

import re
from pathlib import Path

from rengu_flow.registry.model_capabilities import model_capability_registry
from rengu_flow_ui.config_schema import get_sections

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "rengu_flow"

_KEY = r'["\']([A-Za-z0-9_]+)["\']'

# receiver["section"].get("key")  /  receiver["section"]["key"]
# (config["optimizer"].get("gradient_release"), self.config["model"].get("dtype"), ...)
_SECTIONED_RE = re.compile(
    r'\bconfig\s*\[\s*["\'](model|adapter|optimizer|preview)["\']\s*\]\s*'
    r'(?:\.get\(\s*' + _KEY + r'|\[\s*' + _KEY + r'\s*\])'
)

# bare_receiver.get("key")  /  bare_receiver["key"]
# (config.get("blocks_to_swap"), self.model_config["dtype"], training_config.get("epochs"), ...)
_BARE_RE = re.compile(
    r'\b(config|training_config|model_config|adapter_config|preview_cfg)\s*'
    r'(?:\.get\(\s*' + _KEY + r'|\[\s*' + _KEY + r'\s*\])'
)

# receiver-name -> flat-key prefix (matches the mapping in the task spec).
_BARE_PREFIX = {
    "config": "",  # config / self.config / training_config -> top-level
    "training_config": "",
    "model_config": "model.",  # model_config / self.model_config -> "model."
    "adapter_config": "adapter.",  # adapter_config / self.adapter_config -> "adapter."
    "preview_cfg": "preview.",  # preview_cfg -> "preview."
}
_SECTION_PREFIX = {
    "model": "model.",
    "adapter": "adapter.",
    "optimizer": "optimizer.",  # config["optimizer"].get("...") -> "optimizer."
    "preview": "preview.",
}

# Keys the regex genuinely finds as "consumed" but that are intentionally NOT a discrete UI
# field. Every entry needs a one-line reason; do not add an entry just to silence a real gap
# — fix the gap in config_schema.py / model_capabilities.py instead.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "_dataset_config_loaded",  # internal handoff (main.py stashes the loaded dataset TOML here)
        "adapter",  # section/table name, not a leaf key
        "model",  # section/table name, not a leaf key
        "optimizer",  # section/table name, not a leaf key
        "preview",  # section/table name, not a leaf key
        "tracking",  # section/table name, not a leaf key
        "train",  # section/table name, not a leaf key
        "bench",  # section/table name, not a leaf key
        "adapter.alpha",  # rejected by validation; alpha=rank is enforced, never user-settable
        "cache_format",  # rejected legacy key (defaults.py raises if present)
        "pretrained_model_name_or_path",  # vendored diffusers compat key, not a rengu-flow option
        "lr_scheduler_args",  # section name; covered by the lr_scheduler_args.extra_params KV editor
        "optimizer.lr",  # one of many free-form [optimizer] keys covered by optimizer.extra_params
        "max_images",  # dataset TOML key, covered by rengu_flow_ui.dataset_schema (not config_schema)
        "subsample_ratio",  # dataset TOML key, covered by rengu_flow_ui.dataset_schema (not config_schema)
    }
)


def _extract_consumed_keys() -> set[str]:
    consumed: set[str] = set()
    for path in SRC_ROOT.rglob("*.py"):
        parts = path.relative_to(REPO_ROOT).parts
        if "vendor" in parts or "tests" in parts:
            continue
        text = path.read_text(encoding="utf-8")
        for m in _SECTIONED_RE.finditer(text):
            section = m.group(1)
            key = m.group(2) or m.group(3)
            consumed.add(_SECTION_PREFIX[section] + key)
        for m in _BARE_RE.finditer(text):
            receiver = m.group(1)
            key = m.group(2) or m.group(3)
            consumed.add(_BARE_PREFIX[receiver] + key)
    return consumed


def _schema_paths() -> set[str]:
    paths: set[str] = set()
    for section in get_sections():
        for field in section["fields"]:
            paths.add(field["path"])
    return paths


def _model_field_paths() -> set[str]:
    paths: set[str] = set()
    for cap in model_capability_registry.values():
        for spec in cap.model_fields:
            paths.add(spec["path"])
    return paths


def test_every_consumed_config_key_is_exposed_in_the_ui() -> None:
    consumed = _extract_consumed_keys()
    covered = _schema_paths() | _model_field_paths() | ALLOWLIST

    missing = sorted(consumed - covered)
    assert not missing, (
        "Config keys the trainer reads but the UI does not expose: "
        f"{missing}. Add a top-level field in rengu_flow_ui/config_schema.py "
        "get_sections() (bare / 'optimizer.*' keys), a 'model.*' field in the "
        "matching capability's model_fields in rengu_flow/registry/model_capabilities.py, "
        "or — only for genuinely intentional non-UI keys — add a justified entry to "
        "ALLOWLIST in this test."
    )

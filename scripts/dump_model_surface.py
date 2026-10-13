"""Dump the per-model integration surface to deterministic JSON ("golden" harness).

Everything a model type touches outside its own package -- capability descriptor, UI schema and
visibility, preview form, ``set_config_defaults``, validation rules, env-var path map, installer
profiles, preview dispatch -- is serialised per canonical model type to
``tests/golden/model_surface/<type>.json`` (plus ``_global.json`` for the model-independent parts).
``tests/test_model_surface_golden.py`` compares the live dump against those files, so a refactor
that is meant to preserve behavior must leave them byte-identical.

    uv run --no-sync python scripts/dump_model_surface.py            # (re)write the goldens
    uv run --no-sync python scripts/dump_model_surface.py --out DIR  # write elsewhere
    uv run --no-sync python scripts/dump_model_surface.py --check    # exit 1 on any drift

Run it against a clean checkout by putting that tree first on ``PYTHONPATH`` (the dump imports
``rengu_flow`` from wherever Python resolves it; the output dir defaults to this script's repo).

Isolation: no ``jobs.db`` / library DB access (the dataset picker is stubbed), no models are
loaded, preview runners are replaced by recorders, host env vars that influence the output are
cleared for the duration of the dump.

Several observed hooks are private names that exist today (``local_env._MODEL_PATH_ENV``,
``utils.preview._run_*_previews`` ...). If a refactor renames them, update the matching probe in
this file and keep the *observed behavior* identical; that is the point of the golden.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
SURFACE_DIR = GOLDEN_DIR / "model_surface"
LEGACY_DIR = GOLDEN_DIR / "legacy_configs"

GLOBAL_KEY = "_global"

# Field keys copied from schema/preview fields into the semantic UI summary.
_FIELD_KEYS = (
    "type", "label", "description", "help", "doc_path", "default", "options", "option_values",
    "importance", "placeholder", "required", "recommended", "path_expect", "min", "gt", "max",
    "example", "allow_custom", "deepspeed_only", "ui",
)

# Env vars that would leak the host into the dump.
_ENV_CLEAR_PREFIXES = ("RENGU_",)
_ENV_KEEP = {"RENGU_PREFLIGHT"}


# ---------------------------------------------------------------------------------------------
# helpers


def jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-safe values (dtypes/paths -> str, keys -> str)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((jsonable(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def dumps(obj: Any) -> str:
    return json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@contextlib.contextmanager
def _clean_env(**extra: str):
    """Drop RENGU_* host env (keeping RENGU_PREFLIGHT) and set ``extra`` for the block."""
    env = {k: v for k, v in os.environ.items() if not (k.startswith(_ENV_CLEAR_PREFIXES) and k not in _ENV_KEEP)}
    env.setdefault("RENGU_PREFLIGHT", "0")
    env.update(extra)
    with mock.patch.dict(os.environ, env, clear=True):
        yield


def _outcome(fn, *args, **kwargs) -> dict[str, Any]:
    """Run ``fn`` capturing stdout; return {"result": ...} or {"error": "Type: msg"} (+stdout)."""
    buf = io.StringIO()
    out: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            res = fn(*args, **kwargs)
        out["result"] = jsonable(res)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - golden records every failure mode
        out["error"] = f"{type(e).__name__}: {e}"
    text = buf.getvalue()
    if text:
        out["stdout"] = text
    return out


def _field_summary(field: dict[str, Any]) -> dict[str, Any]:
    return {k: jsonable(field[k]) for k in _FIELD_KEYS if k in field}


def _walk_clauses(clause: Any):
    if not isinstance(clause, dict):
        return
    yield clause
    for key in ("all", "any"):
        for sub in clause.get(key, ()) or ():
            yield from _walk_clauses(sub)
    if "not" in clause:
        yield from _walk_clauses(clause["not"])


# ---------------------------------------------------------------------------------------------
# schema (UI)


def _load_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    """``get_schema()`` under RENGU_ENGINE=deepspeed and =accelerate, with the UI library DB
    stubbed out (the dataset picker is the only DB reader)."""
    from rengu_flow_ui import config_schema, datasets_store

    out = []
    for engine in ("deepspeed", "accelerate"):
        with _clean_env(RENGU_ENGINE=engine), mock.patch.object(
            datasets_store, "list_for_training_picker", lambda: []
        ):
            out.append(config_schema.get_schema())
    return out[0], out[1]


def _schema_fields(schema: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"{section['id']}:{field['path']}", field)
        for section in schema["sections"]
        for field in section["fields"]
    ]


def _visible(fields, form, caps) -> list[tuple[str, dict[str, Any]]]:
    """Fields visible for ``form`` as (id, field). Several capabilities may emit the same
    ``section:path`` each with its own gate, so visibility is applied BEFORE indexing; if two
    fields with the same path are visible at once the later ones get ``#2``, ``#3`` ... ids."""
    from rengu_flow_ui.field_visibility import field_visible

    out: list[tuple[str, dict[str, Any]]] = []
    count: dict[str, int] = {}
    for key, field in fields:
        if not field_visible(field, form, caps):
            continue
        count[key] = count.get(key, 0) + 1
        out.append((key if count[key] == 1 else f"{key}#{count[key]}", field))
    return out


def _visible_keys(fields, form, caps) -> list[str]:
    return [key for key, _field in _visible(fields, form, caps)]


def _indexed(fields) -> dict[str, dict[str, Any]]:
    """All schema fields keyed by ``section:path`` (``#n`` suffix for repeated paths)."""
    out: dict[str, dict[str, Any]] = {}
    count: dict[str, int] = {}
    for key, field in fields:
        count[key] = count.get(key, 0) + 1
        out[key if count[key] == 1 else f"{key}#{count[key]}"] = field
    return out


def _trigger_candidates(fields) -> list[tuple[str, Any]]:
    """(form key, value) pairs referenced by any visibility clause, discovered from the schema
    itself so new gates are exercised without editing this list."""
    skip = {"model.type", "_has_adapter", "adapter.type"}
    seen: dict[str, Any] = {}
    for _key, field in fields:
        vis = field.get("visibility")
        for clause in _walk_clauses(vis):
            if "form_nonempty" in clause:
                seen[f"form_nonempty:{clause['form_nonempty']}"] = (clause["form_nonempty"], "1")
            if "form_map_truthy" in clause:
                spec = clause["form_map_truthy"] or {}
                seen[f"map:{spec.get('path')}:{spec.get('key')}"] = (spec.get("path"), {spec.get("key"): True})
            name = clause.get("field")
            if name and name not in skip:
                values = []
                if "equals" in clause:
                    values.append(clause["equals"])
                values.extend(clause.get("in") or [])
                for v in values:
                    seen[f"field:{name}={json.dumps(jsonable(v))}"] = (name, v)
    return [seen[k] for k in sorted(seen)]


def _ui_semantic(type_id: str, cap, schema: dict[str, Any]) -> dict[str, Any]:
    from rengu_flow_ui.field_visibility import model_specific_paths, prune_form_for_model

    caps = schema["registries"]["model_capabilities"]
    fields = _schema_fields(schema)
    details: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()

    def visible_ids(form: dict[str, Any]) -> list[str]:
        ids = []
        for fid, field in _visible(fields, form, caps):
            summary = _field_summary(field)
            if fid in details and details[fid] != summary:
                # Same id, different content across states (e.g. adapter.target_include per
                # adapter kind): keep every variant under a content-hashed id.
                digest = hashlib.sha1(dumps(summary).encode()).hexdigest()[:6]
                conflicts.add(fid)
                fid = f"{fid}~{digest}"
            details.setdefault(fid, summary)
            ids.append(fid)
        return ids

    states: dict[str, dict[str, Any]] = {"no_adapter": {"model.type": type_id, "_has_adapter": False}}
    for kind in cap.adapters:
        states[f"adapter:{kind}"] = {"model.type": type_id, "_has_adapter": True, "adapter.type": kind}
    visible_by_state = {name: visible_ids(form) for name, form in states.items()}

    base_name = "adapter:lora" if "adapter:lora" in states else "no_adapter"
    base_form = states[base_name]
    base_ids = visible_by_state[base_name]
    base_set = set(base_ids)
    deltas: dict[str, Any] = {}
    for fkey, value in _trigger_candidates(fields):
        ids = visible_ids({**base_form, fkey: value})
        plus = [k for k in ids if k not in base_set]
        minus = [k for k in base_ids if k not in set(ids)]
        if plus or minus:
            deltas[f"{fkey}={json.dumps(jsonable(value), sort_keys=True)}"] = {"show": plus, "hide": minus}

    duplicates = sorted(
        {fid.split("#")[0] for ids in visible_by_state.values() for fid in ids if "#" in fid.split("~")[0]}
    )

    # Fields of the other-model-specific paths that stay pruned for this model.
    pruned_form = prune_form_for_model(
        {p: "x" for owned in model_specific_paths().values() for p in owned} | {"model.type": type_id}
    )

    return {
        "base_state_for_deltas": base_name,
        "states": visible_by_state,
        "trigger_deltas": deltas,
        "fields": {k: details[k] for k in sorted(details)},
        "duplicate_visible_paths": duplicates,
        "field_ids_with_per_state_variants": sorted(conflicts),
        "owned_model_paths": sorted(model_specific_paths().get(type_id, ())),
        "prune_form_for_model_keeps": sorted(pruned_form),
    }


def _ui_schema_slice(type_id: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Raw schema fields whose gate mentions ``type_id`` or a capability feature the model
    has (``{"capability": f}`` clauses / ``when_capability``), full dicts, schema order."""
    from rengu_flow.registry.model_capabilities import capabilities_for_api

    features = {f for f, on in (capabilities_for_api()[type_id].get("features") or {}).items() if on}
    out = []
    for section in schema["sections"]:
        for field in section["fields"]:
            gate = json.dumps(jsonable({k: field.get(k) for k in ("visibility", "when", "when_capability")}))
            names_type = f'"{type_id}"' in gate
            names_feature = any(
                f'"capability": "{f}"' in gate or f'"when_capability": "{f}"' in gate for f in features
            )
            if names_type or names_feature:
                out.append({"section": section["id"], **jsonable(field)})
    return out


def _preview_entry(type_id: str, caps: dict[str, Any]) -> dict[str, Any]:
    from rengu_flow_ui.field_visibility import field_visible
    from rengu_flow_ui.preview_form import get_preview_entry_fields

    form = {"model.type": type_id}
    visible = [f for f in get_preview_entry_fields() if field_visible(f, form, caps)]
    return {"visible": jsonable(visible)}


# ---------------------------------------------------------------------------------------------
# capability / validation / defaults


def _capability_section(type_id: str, cap) -> dict[str, Any]:
    from rengu_flow.registry.model_capabilities import capabilities_for_api

    extra = {
        name: jsonable(getattr(cap, name, "<absent>"))
        for name in ("transformer_blocks", "text_cache_keys", "text_cache_version", "text_cache_files")
    }
    return {"api": jsonable(capabilities_for_api()[type_id]), "non_api_attributes": extra}


def _text_cache(cap) -> dict[str, Any]:
    keys = dict(getattr(cap, "text_cache_keys", {}) or {})
    files = list(getattr(cap, "text_cache_files", []) or [])
    cases: dict[str, dict[str, Any]] = {"empty": {}}
    for key, default in keys.items():
        cases[f"{key}=non-default"] = {key: 2 if not isinstance(default, int) or default is None else default + 1}
        cases[f"{key}=default"] = {key: default}
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        f1 = tmpdir / "enc.safetensors"
        f1.write_bytes(b"1234567")
        d1 = tmpdir / "enc_dir"
        d1.mkdir()
        (d1 / "a.bin").write_bytes(b"12")
        for key in files:
            cases[f"{key}=missing-file"] = {key: str(tmpdir / "nope.safetensors")}
            cases[f"{key}=file"] = {key: str(f1)}
            cases[f"{key}=folder"] = {key: str(d1)}
        if len(files) > 1:
            cases["files=all-first-wins"] = {key: str(f1) for key in files}
        for name, model_cfg in sorted(cases.items()):
            res = _outcome(cap.text_cache_identity, {"type": cap.type_id, **model_cfg})
            out[name] = _normalize_identity(res, tmpdir)
    return out


def _normalize_identity(res: dict[str, Any], tmpdir: Path) -> dict[str, Any]:
    ident = res.get("result")
    if isinstance(ident, dict):
        norm = {}
        for k, v in ident.items():
            if isinstance(v, str):
                v = re.sub(r"^([\w]+):(\d+):\d+$", r"\1:\2:<mtime_ns>", v)
                v = re.sub(r"^([\w]+):[0-9a-f]{16}$", r"\1:<dir-digest>", v)
            norm[k] = v
        res = {**res, "result": norm}
    return res


def _base_model(type_id: str, cap) -> dict[str, Any]:
    from rengu_flow.registry.model_config_rules import one_of_groups, required_model_keys

    model: dict[str, Any] = {"type": type_id, "dtype": "bfloat16"}
    for key in required_model_keys(cap):
        model[key] = f"placeholder/{key}"
    for group in one_of_groups(cap):
        if not any(k in model for k in group):
            model[group[0]] = f"placeholder/{group[0]}"
    return model


def _validation_section(type_id: str, cap) -> dict[str, Any]:
    from rengu_flow.config.validation import collect_validation_errors
    from rengu_flow.registry.model_config_rules import (
        FEATURE_GATED_TRAINING_KEYS,
        one_of_groups,
        required_model_keys,
        validate_config_model_rules,
    )

    base_model = _base_model(type_id, cap)
    base = {
        "dataset": "examples/minimal_dataset.toml",
        "model": base_model,
        "optimizer": {"type": "adamw", "lr": 1e-4},
    }
    cases: dict[str, dict[str, Any]] = {
        "type_only": {"model": {"type": type_id}},
        "base_ok": base,
        "type_uppercase": {"model": {**base_model, "type": type_id.upper()}},
    }
    for group in one_of_groups(cap):
        for key in group:
            cases[f"one_of_only:{key}"] = {
                "model": {**{k: v for k, v in base_model.items() if k not in {g for gr in one_of_groups(cap) for g in gr}}, key: "x"}
            }
    for key, feature in sorted(FEATURE_GATED_TRAINING_KEYS.items()):
        val: Any = {"drop_ratio": 0.5} if key == "tread" else 4
        cases[f"gated:{key}"] = {**base, key: val}
    cases["tread:bad_ratio"] = {**base, "tread": {"drop_ratio": 2}}
    cases["tread:no_ratio"] = {**base, "tread": {"start_block": 1}}
    cases["tread:bad_route"] = {**base, "tread": {"drop_ratio": 0.5, "start_block": 5, "end_block": 2}}
    cases["tread:not_table"] = {**base, "tread": 3}
    for spec in cap.model_fields:
        key = spec["path"].split(".", 1)[-1]
        if spec.get("options"):
            cases[f"field_bad_option:{key}"] = {**base, "model": {**base_model, key: "__bad__"}}
        if spec.get("type") in ("integer", "number") and ("min" in spec or "gt" in spec):
            bound = spec.get("min", spec.get("gt"))
            cases[f"field_below_bound:{key}"] = {**base, "model": {**base_model, key: bound - 1}}
            cases[f"field_non_integer:{key}"] = {**base, "model": {**base_model, key: bound + 0.5}}
    out = {
        "required_model_keys": required_model_keys(cap),
        "one_of_groups": one_of_groups(cap),
        "rules": {},
        "collect_validation_errors": {},
    }
    for name, cfg in sorted(cases.items()):
        out["rules"][name] = _outcome(validate_config_model_rules, copy.deepcopy(cfg))
    for name in ("type_only", "base_ok"):
        out["collect_validation_errors"][name] = _outcome(collect_validation_errors, copy.deepcopy(cases[name]))
    return out


def _normalize_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = jsonable(cfg)
    expected = min(8, os.cpu_count() or 1)
    if cfg.get("cache_num_proc") == expected:
        cfg["cache_num_proc"] = "<min(8,cpu_count)>"
    return cfg


def _run_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    from rengu_flow.config.defaults import set_config_defaults

    work = copy.deepcopy(cfg)
    res = _outcome(set_config_defaults, work)
    out: dict[str, Any] = {}
    if "error" in res:
        out["error"] = res["error"]
    out["config"] = _normalize_defaults(work)
    if "stdout" in res:
        out["stdout"] = res["stdout"]
    return out


def _defaults_section(type_id: str, cap) -> dict[str, Any]:
    base_model = _base_model(type_id, cap)

    def cfg(model_extra=None, **top) -> dict[str, Any]:
        out = {"model": {**base_model, **(model_extra or {})}, "optimizer": {"type": "adamw", "lr": 1e-4}}
        out.update(copy.deepcopy(top))
        return out

    preview = {"prompts": ["a prompt"]}
    cases: dict[str, dict[str, Any]] = {
        "plain": cfg(),
        "no_preview_key": cfg(),
        "preview_empty_table": cfg(preview={}),
        "preview_prompts": cfg(preview=preview),
        "preview_prompts_overrides": cfg(
            preview={**preview, "num_inference_steps": 7, "guidance_scale": 1.5, "width": 512, "preview_offload_text_encoder": False}
        ),
        "preview_not_a_table": cfg(preview="oops"),
        "fp8": cfg({"transformer_fp8_matmul": True}),
        "fp8_e4m3": cfg({"transformer_fp8_matmul": True, "fp8_matmul_dtype": "e4m3"}),
        "fp8_bad_matmul_dtype": cfg({"transformer_fp8_matmul": True, "fp8_matmul_dtype": "bogus"}),
        "fp8_grad_fp8": cfg({"fp8_grad_mode": "fp8"}),
        "fp8_grad_bad": cfg({"fp8_grad_mode": "bogus"}),
        "q4bit": cfg({"transformer_4bit": True}),
        "fp8_and_4bit": cfg({"transformer_fp8_matmul": True, "transformer_4bit": True}),
        "q4bit_blocks_to_swap": cfg({"transformer_4bit": True}, blocks_to_swap=8),
        "q4bit_blocks_to_swap_explicit_non_reentrant": cfg(
            {"transformer_4bit": True}, blocks_to_swap=8, reentrant_activation_checkpointing=False
        ),
        "fp8_compile_block_ac": cfg(
            {"transformer_fp8_matmul": True}, compile=True, compile_scope="block", activation_checkpointing=True
        ),
        "fp8_compile_model_ac": cfg(
            {"transformer_fp8_matmul": True}, compile=True, compile_scope="model", activation_checkpointing=True
        ),
        "fp8_compile_block_ac_explicit_non_reentrant": cfg(
            {"transformer_fp8_matmul": True},
            compile=True,
            compile_scope="block",
            activation_checkpointing=True,
            reentrant_activation_checkpointing=False,
        ),
        "ac_selective": cfg(activation_checkpointing="selective"),
        "ac_unsloth": cfg(activation_checkpointing="unsloth"),
        "ac_auto": cfg(activation_checkpointing="auto"),
        "diffusion_model_dtype_bf16": cfg({"diffusion_model_dtype": "bfloat16"}),
        "diffusion_model_dtype_float8": cfg({"diffusion_model_dtype": "float8_e4m3fn"}),
        "transformer_dtype_fp16": cfg({"transformer_dtype": "float16"}),
        "diffusion_and_transformer_dtype": cfg({"diffusion_model_dtype": "float8", "transformer_dtype": "float16"}),
        "save_dtype": cfg(save_dtype="float16"),
        "micro_batch_dict": cfg(micro_batch_size_per_gpu={"512": 2, "1024": 1}),
        "model_dtype_bad": {"model": {**base_model, "dtype": "bf16"}},
    }
    for kind in cap.adapters:
        cases[f"adapter:{kind}"] = cfg(adapter={"type": kind, "rank": 4})
    cases["adapter:lora_dim_alias"] = cfg(adapter={"type": "lora", "dim": 8})
    cases["adapter:lora_rank_and_dim"] = cfg(adapter={"type": "lora", "rank": 4, "dim": 8})
    cases["adapter:lora_alpha_rejected"] = cfg(adapter={"type": "lora", "rank": 4, "alpha": 2})
    cases["adapter:lora_dtype_bf16"] = cfg(adapter={"type": "lora", "rank": 4, "dtype": "bfloat16"})
    cases["adapter:unknown_type"] = cfg(adapter={"type": "bogus", "rank": 4})
    cases["adapter:unknown_lycoris"] = cfg(adapter={"type": "lycoris_bogus", "rank": 4})
    cases["adapter:lora_with_preview_and_fp8"] = cfg(
        {"transformer_fp8_matmul": True}, adapter={"type": "lora", "rank": 4}, preview=copy.deepcopy(preview)
    )
    runs = {name: _run_defaults(c) for name, c in sorted(cases.items())}
    # Store ``plain`` in full and every other case as a flat dotted-path diff against it: the
    # defaults table is ~200 keys, and a full copy per case made the goldens unreadable.
    plain = _flatten(runs["plain"]["config"])
    out: dict[str, Any] = {"plain": runs["plain"]}
    for name, run in runs.items():
        if name == "plain":
            continue
        flat = _flatten(run["config"])
        entry: dict[str, Any] = {k: v for k, v in run.items() if k != "config"}
        entry["set"] = {k: v for k, v in sorted(flat.items()) if k not in plain or plain[k] != v}
        entry["unset"] = sorted(k for k in plain if k not in flat)
        out[name] = entry
    return out


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted-path view of nested dicts; lists and empty dicts are leaves."""
    if isinstance(obj, dict) and obj:
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: obj}


# ---------------------------------------------------------------------------------------------
# env vars / installer / preview dispatch


def _env_section(type_id: str, cap) -> dict[str, Any]:
    from rengu_flow.config import local_env

    mapping = dict(getattr(local_env, "_MODEL_PATH_ENV", {}).get(type_id, {}))
    out: dict[str, Any] = {
        "env_map": mapping,
        "_notes": ["env map read from local_env._MODEL_PATH_ENV (private); update the probe if it moves"],
    }
    cases: dict[str, dict[str, Any]] = {
        "no_model_table": {},
        "type_only": {"model": {"type": type_id}},
        "unknown_type": {"model": {"type": "no_such_model"}},
    }
    keys = sorted(mapping)
    cases["all_set_missing_files"] = {"model": {"type": type_id, **{k: f"nonexistent/{k}.bin" for k in keys}}}
    for key in keys:
        cases[f"only:{key}"] = {"model": {"type": type_id, key: f"nonexistent/{key}.bin"}}
        cases[f"blank:{key}"] = {"model": {"type": type_id, key: "   "}}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        files = {}
        for key in keys:
            p = tmpdir / f"{key}.bin"
            p.write_bytes(b"x")
            files[key] = str(p)
        folders = {}
        for key in keys:
            d = tmpdir / f"{key}_dir"
            d.mkdir()
            folders[key] = str(d)
        cases["all_set_existing_files"] = {"model": {"type": type_id, **files}}
        cases["all_set_existing_folders"] = {"model": {"type": type_id, **folders}}
        for key in keys:
            cases[f"only_existing_file:{key}"] = {"model": {"type": type_id, key: files[key]}}
            cases[f"only_existing_folder:{key}"] = {"model": {"type": type_id, key: folders[key]}}
        out["model_path_errors"] = {
            name: _outcome(local_env.model_path_errors, copy.deepcopy(c))
            for name, c in sorted(cases.items())
        }
        out["model_path_errors"] = json.loads(
            json.dumps(out["model_path_errors"]).replace(str(tmpdir).replace("\\", "\\\\"), "<TMP>").replace(str(tmpdir), "<TMP>")
        )
    # apply_model_paths_from_env: with the env var set / unset / blank
    applied = {}
    for state, env in (
        ("env_all_set", {name: f"/env/{name}" for name in mapping.values()}),
        ("env_blank", {name: "" for name in mapping.values()}),
        ("env_unset", {}),
    ):
        model = {"type": type_id, **{k: f"orig/{k}" for k in keys}}
        with _clean_env(**env):
            cfg = {"model": model}
            names = local_env.apply_model_paths_from_env(cfg)
        applied[state] = {"applied": names, "model": cfg["model"]}
    out["apply_model_paths_from_env"] = jsonable(applied)
    return out


def _installer_section(type_id: str, cap) -> dict[str, Any]:
    from rengu_flow.install.manager import profiles_for_config_dict

    model = {"type": type_id}
    cases: dict[str, dict[str, Any]] = {
        "model_only": {"model": model},
        "model_type_uppercase": {"model": {"type": type_id.upper()}},
        "no_model": {},
        "no_adapter_adamw": {"model": model, "optimizer": {"type": "adamw"}},
    }
    for kind in cap.adapters:
        cases[f"adapter:{kind}"] = {"model": model, "adapter": {"type": kind}}
    for otype in ("adamw8bit", "prodigy", "adamw_optimi", "kaon.adakaon", "adakaon", "nekaon", "lion"):
        cases[f"optimizer:{otype}"] = {"model": model, "adapter": {"type": "lora"}, "optimizer": {"type": otype}}
    cases["lokr+adamw8bit"] = {"model": model, "adapter": {"type": "lokr"}, "optimizer": {"type": "adamw8bit"}}
    return {name: _outcome(profiles_for_config_dict, c) for name, c in sorted(cases.items())}


class _FakeOptimizer:
    def __init__(self, log: list[str]):
        self._log = log

    def eval(self) -> None:
        self._log.append("optimizer.eval")

    def train(self) -> None:
        self._log.append("optimizer.train")


def _fake_model(name: str, log: list[str]):
    class FakeModel:
        def prepare_block_swap_inference(self, disable_block_swap=False):
            log.append(f"prepare_block_swap_inference(disable={disable_block_swap})")

        def prepare_block_swap_training(self):
            log.append("prepare_block_swap_training")

        def restore_after_preview(self):
            log.append("restore_after_preview")

    m = FakeModel()
    m.name = name
    return m


def _preview_dispatch(model_name: str) -> dict[str, Any]:
    """Which preview runner ``run_previews`` picks for ``model_name`` (runners are recorders)."""
    from rengu_flow.utils import preview

    prompts = {"prompts": ["p0"]}
    cases: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "single_stage": ({"preview": prompts, "pipeline_stages": 1}, {}),
        "two_stages": ({"preview": prompts, "pipeline_stages": 2}, {}),
        "no_pipeline_stages_key": ({"preview": prompts}, {}),
        "no_prompts": ({"preview": {}}, {}),
        "disable_block_swap": ({"preview": prompts}, {"disable_block_swap": True}),
        "with_optimizer": ({"preview": prompts}, {"optimizer": True}),
        "control_images": (
            {"preview": {"prompts": [{"prompt": "p0", "control_images": ["a.png"]}]}},
            {},
        ),
        "runner_raises": ({"preview": prompts}, {"raise": True}),
    }
    out: dict[str, Any] = {}
    for case, (config, opts) in sorted(cases.items()):
        log: list[str] = []

        def recorder(label):
            def run(model, preview_cfg, prompts_, sink, step):
                log.append(f"runner:{label}")
                if opts.get("raise"):
                    raise RuntimeError("boom")

            return run

        model = _fake_model(model_name, log)
        optimizer = _FakeOptimizer(log) if opts.get("optimizer") else None
        buf = io.StringIO()
        with mock.patch.object(preview, "_run_sdxl_previews", recorder("sdxl")), mock.patch.object(
            preview, "_run_cosmos_previews", recorder("dit")
        ), mock.patch.object(preview, "_dist_barrier", lambda: log.append("barrier")), mock.patch.object(
            preview, "is_main_process", lambda: True
        ), mock.patch.object(preview, "empty_cuda_cache", lambda: None), contextlib.redirect_stdout(
            buf
        ), contextlib.redirect_stderr(io.StringIO()):
            try:
                preview.run_previews(
                    model,
                    copy.deepcopy(config),
                    sink=object(),
                    step=5,
                    disable_block_swap=bool(opts.get("disable_block_swap")),
                    optimizer=optimizer,
                )
                err = None
            except (Exception, SystemExit) as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
        text = re.sub(r"Preview complete in [\d.]+s", "Preview complete in <t>s", buf.getvalue())
        entry: dict[str, Any] = {"calls": log, "stdout": text}
        if err:
            entry["error"] = err
        out[case] = entry
    return out


# ---------------------------------------------------------------------------------------------
# legacy config corpus


def run_legacy_config(path: Path) -> dict[str, Any]:
    """Run every config-side check on one legacy TOML and record the outcomes."""
    import toml

    from rengu_flow.config import validation as v
    from rengu_flow.config.local_env import model_path_errors
    from rengu_flow.install.manager import profiles_for_config_dict
    from rengu_flow.registry.model_config_rules import validate_config_model_rules

    with open(path, encoding="utf-8") as f:
        cfg = json.loads(json.dumps(toml.load(f)))
    out: dict[str, Any] = {
        "collect_validation_errors": _outcome(v.collect_validation_errors, copy.deepcopy(cfg)),
        "collect_default_prerequisite_issues": _outcome(v.collect_default_prerequisite_issues, copy.deepcopy(cfg)),
        "collect_misplaced_top_level_keys": _outcome(v.collect_misplaced_top_level_keys, copy.deepcopy(cfg)),
        "collect_cadence_issues": _outcome(v.collect_cadence_issues, copy.deepcopy(cfg)),
        "validate_config_model_rules": _outcome(validate_config_model_rules, copy.deepcopy(cfg)),
        "model_path_errors": _outcome(model_path_errors, copy.deepcopy(cfg)),
        "profiles_for_config_dict": _outcome(profiles_for_config_dict, copy.deepcopy(cfg)),
    }
    defaults = _run_defaults(cfg)
    out["set_config_defaults"] = defaults
    if "error" not in defaults:
        post = copy.deepcopy(cfg)
        _outcome(__import__("rengu_flow.config.defaults", fromlist=["x"]).set_config_defaults, post)
        out["collect_validation_warnings_after_defaults"] = _outcome(v.collect_validation_warnings, post)
    return out


def dump_legacy(legacy_dir: Path = LEGACY_DIR) -> dict[str, Any]:
    with _clean_env():
        return {p.name: run_legacy_config(p) for p in sorted(legacy_dir.glob("*.toml"))}


# ---------------------------------------------------------------------------------------------
# assembling


def canonical_types() -> list[str]:
    from rengu_flow.registry.model_capabilities import get_canonical_model_types

    return get_canonical_model_types()


def _global_section(schema_ds: dict[str, Any], schema_acc: dict[str, Any]) -> dict[str, Any]:
    from rengu_flow.install import profiles
    from rengu_flow.registry import models
    from rengu_flow.registry.model_capabilities import (
        ADAPTER_FIELD_TEMPLATES,
        DEFAULT_ADAPTER_LABELS,
        model_capability_registry,
        normalize_model_type,
    )
    from rengu_flow.registry.model_config_rules import FEATURE_GATED_TRAINING_KEYS
    from rengu_flow_ui import preview_form
    from rengu_flow_ui.field_visibility import all_model_specific_paths

    ds = _indexed(_schema_fields(schema_ds))
    acc = _indexed(_schema_fields(schema_acc))
    registries = {k: v for k, v in schema_ds["registries"].items() if k not in ("model_capabilities", "dataset_paths")}
    schema_body = {
        "sections": schema_ds["sections"],
        "default_new_config_toml": schema_ds["default_new_config_toml"],
        "host": {k: v for k, v in schema_ds["host"].items() if k != "is_windows"},
    }
    aliases = {a.lower(): normalize_model_type(a) for cap in model_capability_registry.values() for a in cap.aliases}
    with _clean_env():
        dispatch_unregistered = {name: _preview_dispatch(name) for name in ("anima", "not_a_model")}
    return {
        "_notes": [
            "dataset_paths registry stubbed to [] (library DB is not touched)",
            "schema_deepspeed = get_schema() under RENGU_ENGINE=deepspeed; accelerate only as delta",
            "registries.model_capabilities is covered per type; host.is_windows is dropped (host dependent)",
        ],
        "canonical_types": canonical_types(),
        "aliases": aliases,
        "builtin_model_modules": jsonable(getattr(models, "_BUILTIN_MODEL_MODULES", "<absent>")),
        "adapter_labels": DEFAULT_ADAPTER_LABELS,
        "adapter_field_templates": ADAPTER_FIELD_TEMPLATES,
        "feature_gated_training_keys": FEATURE_GATED_TRAINING_KEYS,
        "all_model_specific_paths": sorted(all_model_specific_paths()),
        "registries": registries,
        "schema_deepspeed": schema_body,
        "schema_accelerate_delta": {
            "removed_field_keys": sorted(set(ds) - set(acc)),
            "added_field_keys": sorted(set(acc) - set(ds)),
            "common_fields_equal": all(jsonable(ds[k]) == jsonable(acc[k]) for k in set(ds) & set(acc)),
            "registries_equal": jsonable({k: v for k, v in schema_acc["registries"].items() if k != "dataset_paths"})
            == jsonable({k: v for k, v in schema_ds["registries"].items() if k != "dataset_paths"}),
            "host": {k: v for k, v in schema_acc["host"].items() if k != "is_windows"},
        },
        "preview_form": {
            "entry_fields": preview_form.get_preview_entry_fields(),
            "global_paths": sorted(preview_form.PREVIEW_GLOBAL_PATHS),
        },
        "install_profiles": {
            "extras": profiles.PROFILE_EXTRAS,
            "import_checks": profiles.PROFILE_IMPORT_CHECKS,
            "git_requirements": profiles.PROFILE_GIT_REQUIREMENTS,
            "all_profile_names": list(profiles.ALL_PROFILE_NAMES),
        },
        "preview_dispatch_unregistered_names": dispatch_unregistered,
    }


def dump_all() -> dict[str, dict[str, Any]]:
    """{type_id: surface, "_global": ..., } -- pure data, JSON-safe."""
    from rengu_flow.registry.model_capabilities import get_capability

    schema_ds, schema_acc = _load_schemas()
    caps = schema_ds["registries"]["model_capabilities"]
    out: dict[str, dict[str, Any]] = {}
    with _clean_env():
        for type_id in canonical_types():
            cap = get_capability(type_id)
            out[type_id] = jsonable(
                {
                    "_notes": [
                        "ui_semantic is evaluated on the deepspeed schema (superset of accelerate)",
                        "ui_schema_slice = raw schema fields whose gate names this type",
                        "preview_dispatch uses recorder runners; no model is loaded",
                    ],
                    "type": type_id,
                    "capability": _capability_section(type_id, cap),
                    "text_cache_identity": _text_cache(cap),
                    "ui_semantic": _ui_semantic(type_id, cap, schema_ds),
                    "ui_schema_slice": _ui_schema_slice(type_id, schema_ds),
                    "preview_entry_form": _preview_entry(type_id, caps),
                    "validation": _validation_section(type_id, cap),
                    "defaults": _defaults_section(type_id, cap),
                    "env": _env_section(type_id, cap),
                    "installer": _installer_section(type_id, cap),
                    "preview_dispatch": _preview_dispatch(type_id),
                }
            )
        out[GLOBAL_KEY] = jsonable(_global_section(schema_ds, schema_acc))
    return out


def write_goldens(
    out_dir: Path = SURFACE_DIR, legacy_dir: Path = LEGACY_DIR, legacy_out: Path | None = None
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in dump_all().items():
        p = out_dir / f"{name}.json"
        p.write_text(dumps(data), encoding="utf-8", newline="\n")
        written.append(p)
    expected = legacy_out or (legacy_dir / "expected.json")
    if legacy_dir.is_dir():
        expected.write_text(dumps(dump_legacy(legacy_dir)), encoding="utf-8", newline="\n")
        written.append(expected)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=SURFACE_DIR, help="model_surface output dir")
    ap.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if the goldens differ")
    args = ap.parse_args(argv)
    if args.check:
        bad = []
        for name, data in dump_all().items():
            p = args.out / f"{name}.json"
            if not p.is_file() or p.read_text(encoding="utf-8") != dumps(data):
                bad.append(str(p))
        if bad:
            print("golden drift:", *bad, sep="\n  ")
            return 1
        return 0
    import rengu_flow

    print(f"dumping from {Path(rengu_flow.__file__).resolve().parent.parent}")
    # A custom --out must never overwrite the committed legacy golden: it goes next to --out.
    legacy_out = None if args.out == SURFACE_DIR else args.out / "legacy_expected.json"
    for p in write_goldens(args.out, args.legacy_dir, legacy_out):
        print(f"wrote {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

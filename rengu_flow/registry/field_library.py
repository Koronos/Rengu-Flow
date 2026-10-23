"""Factories for the ``model_fields`` entries that recur across model specs (torch-free).

A DiT spec repeats the same structural entries (component paths, the diffusers bundle folder,
``transformer_dtype``, ``shift``, ``sigmoid_scale``, ``timestep_sample_method``, the frozen-base
quantization knobs). The factories hold what is common (path, label, type, options, default,
visibility); each spec passes its own ``description`` / ``help`` / ``doc`` / ``placeholder`` as
overrides, so per-model wording stays in the model's spec (docs/spec/model-spec-refactor.md,
phase 2c; harmonizing the texts is the optional phase 6).

Every factory returns a fresh dict, so a spec (or a consumer) mutating its copy never leaks into
another model. ``help`` / ``doc`` are the long form text and doc link of the field: the UI schema
reads them from the field (``rengu_flow_ui/config_schema.py``), with ``doc`` defaulting to the
spec's ``user_doc`` whenever a field has ``help``.
"""

from __future__ import annotations

from typing import Any

from rengu_flow.registry.model_spec import QuantSpec

TRANSFORMER_DTYPE_DESCRIPTION = "DiT checkpoint load only; defaults to Model dtype. VAE/text unchanged."
BUNDLE_FOLDER_LABEL = "Diffusers folder (alternative)"


def _entry(base: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """``base`` plus every override that is not None (None = "not given")."""
    out = dict(base)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def component_path(
    key: str,
    label: str,
    description: str,
    *,
    placeholder: str | None = None,
    path_expect: str | None = None,
    importance: str | None = None,
    required: bool | None = None,
    show_if_set: bool | None = None,
    help: str | None = None,
    doc: str | None = None,
) -> dict[str, Any]:
    """``model.<key>``: a weights file/folder path field."""
    return _entry(
        {"path": f"model.{key}", "label": label, "type": "path"},
        path_expect=path_expect,
        required=required,
        importance=importance,
        placeholder=placeholder,
        show_if_set=show_if_set,
        description=description,
        help=help,
        doc=doc,
    )


def bundle_folder(
    key: str,
    description: str,
    *,
    label: str = BUNDLE_FOLDER_LABEL,
    placeholder: str | None = None,
    help: str | None = None,
    doc: str | None = None,
) -> dict[str, Any]:
    """``model.<key>``: the whole-bundle diffusers folder that fills any component left empty
    (``ModelCapability.bundle_key``)."""
    return _entry(
        {"path": f"model.{key}", "label": label, "type": "path", "path_expect": "dir", "importance": "recommended"},
        placeholder=placeholder,
        description=description,
        help=help,
        doc=doc,
    )


def transformer_dtype(
    *, description: str = TRANSFORMER_DTYPE_DESCRIPTION, help: str | None = None, doc: str | None = None
) -> dict[str, Any]:
    """``model.transformer_dtype``: the DiT checkpoint load dtype."""
    return _entry(
        {
            "path": "model.transformer_dtype",
            "label": "Main model load dtype",
            "type": "select",
            "options_key": "dtypes",
        },
        description=description,
        help=help,
        doc=doc,
    )


def diffusion_model_dtype(
    description: str,
    *,
    label: str = "DiT forward dtype",
    ui: bool | None = None,
    help: str | None = None,
    doc: str | None = None,
) -> dict[str, Any]:
    """``model.diffusion_model_dtype``: the forward autocast dtype (``label`` names the network:
    "UNet forward dtype" for sdxl). ``ui=False`` keeps it TOML-only."""
    return _entry(
        {"path": "model.diffusion_model_dtype", "label": label, "type": "select", "options_key": "dtypes"},
        ui=ui,
        description=description,
        help=help,
        doc=doc,
    )


def shift(
    description: str,
    *,
    placeholder: str,
    help: str | None = None,
    doc: str | None = None,
) -> dict[str, Any]:
    """``model.shift``: fixed timestep shift (> 0); what "empty" means differs per model."""
    return _entry(
        {"path": "model.shift", "label": "Fixed timestep shift", "type": "number", "gt": 0},
        placeholder=placeholder,
        description=description,
        help=help,
        doc=doc,
    )


def sigmoid_scale(description: str, *, help: str | None = None, doc: str | None = None) -> dict[str, Any]:
    """``model.sigmoid_scale``: scale on the logit-normal sample before the sigmoid."""
    return _entry(
        {"path": "model.sigmoid_scale", "label": "Logit-normal sigmoid scale", "type": "number", "default": 1.0},
        description=description,
        help=help,
        doc=doc,
    )


def timestep_sample_method(
    description: str,
    *,
    options: tuple[str, ...] | None = None,
    default: str | None = None,
    help: str | None = None,
    doc: str | None = None,
) -> dict[str, Any]:
    """``model.timestep_sample_method``: ``logit_normal`` (default) or ``uniform``; a model with its
    own reference sampler adds it to ``options`` (listed first or last as it wants) and may make it
    the ``default`` (the model's pipeline implements it, and must also list it in ``model_defaults``)."""
    return _entry(
        {
            "path": "model.timestep_sample_method",
            "label": "Timestep sample method",
            "type": "select",
            "options": list(options) if options else ["logit_normal", "uniform"],
            "default": default or "logit_normal",
        },
        description=description,
        help=help,
        doc=doc,
    )


_FP8_LABELS = {
    "tensorwise": "Quantize base to fp8 (tensorwise scaled matmul)",
    "scaled_matmul": "Quantize base to fp8 (scaled matmul)",
}


def quant_fields(
    quant: QuantSpec,
    *,
    nf4: dict[str, str] | None = None,
    fp8: dict[str, str] | None = None,
    grad_mode: dict[str, str] | None = None,
    fp8_dtype: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """The frozen-base quantization knobs ``quant`` enables, keyed ``nf4`` / ``fp8`` /
    ``grad_mode`` / ``fp8_dtype`` (the spec places each where it wants in its field order; the
    order is the form order). Each keyword is that field's ``description`` / ``help`` / ``doc``
    overrides."""
    out: dict[str, dict[str, Any]] = {}
    if quant.nf4:
        out["nf4"] = _entry(
            {
                "path": "model.transformer_4bit",
                "label": "Quantize base to 4-bit (NF4)",
                "type": "boolean",
                "default": False,
                "when_model_has_adapter": True,
            },
            **(nf4 or {}),
        )
    if quant.fp8:
        out["fp8"] = _entry(
            {
                "path": "model.transformer_fp8_matmul",
                "label": _FP8_LABELS[quant.fp8],
                "type": "boolean",
                "default": False,
                "when_model_has_adapter": True,
            },
            **(fp8 or {}),
        )
    if quant.fp8 and quant.grad_mode:
        out["grad_mode"] = _entry(
            {
                "path": "model.fp8_grad_mode",
                "label": "fp8 backward gradient precision",
                "type": "select",
                "options": ["bf16", "fp8"],
                "default": "bf16",
                "visibility": {
                    "all": [
                        {"when_model_has_adapter": True},
                        {"field": "model.transformer_fp8_matmul", "equals": True},
                    ],
                },
            },
            **(grad_mode or {}),
        )
    if quant.fp8 == "scaled_matmul" and quant.fp8_dtypes:
        out["fp8_dtype"] = _entry(
            {
                "path": "model.fp8_matmul_dtype",
                "label": "fp8 weight format",
                "type": "select",
                "options": list(quant.fp8_dtypes),
                "default": quant.fp8_dtypes[0],
                "show_if_set": True,
            },
            **(fp8_dtype or {}),
        )
    return out

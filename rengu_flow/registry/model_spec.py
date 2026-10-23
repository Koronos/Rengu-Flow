"""Model descriptor types: ``ModelCapability`` and the value objects a model spec fills in.

One model = one package + one descriptor: each model package defines ``SPEC = ModelCapability(...)``
in its ``spec.py`` (sdxl: ``rengu_flow/model/sdxl_spec.py``) and ``registry/builtin_models.py`` lists
those modules; ``registry/model_capabilities.py`` registers them. See
``docs/spec/model-spec-refactor.md``.

Torch-free on purpose (stdlib + ``networks.lycoris_meta`` only): config validation, defaults, the
installer and the web UI schema read specs without importing torch or the model code. The spec
types live here rather than in ``model_capabilities`` so a spec module can be imported first
without re-entering the registration loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Display labels for adapter kinds. Convention (matches the optimizer registry):
# anything from an external library is "Vendor.Name"; rengu's own implementations
# carry no prefix. LoRA comes from PEFT; the lycoris_* family from lycoris-lora;
# the built-in `lokr` is rengu's own vendored implementation (no prefix).
DEFAULT_ADAPTER_LABELS: dict[str, str] = {
    "lora": "Peft.LoRA",
    "lokr": "LoKr",
    "lycoris_locon": "Lycoris.LoCon",
    "lycoris_loha": "Lycoris.LoHa",
    "lycoris_lokr": "Lycoris.LoKr",
    "lycoris_dylora": "Lycoris.DyLoRA",
    "lycoris_glora": "Lycoris.GLoRA",
    "lycoris_diag_oft": "Lycoris.DiagOFT",
    "lycoris_boft": "Lycoris.BOFT",
}

# Knobs that only act on Conv modules (train_conv, use_tucker) or on affine LayerNorm/GroupNorm
# (train_norm) carry "unless_capability": the schema hides them on models whose adapted network
# has neither (features[LINEAR_ONLY_ADAPTERS]) — there they are no-ops (or, train_norm, a
# startup error).
LINEAR_ONLY_ADAPTERS = "linear_only_adapters"

# Text-encoder outputs every model caches as ragged (variable-length) rows; a spec's
# ``ragged_cache_keys`` adds its model-specific ones (see utils/cache._SEQUENCE_TENSOR_KEYS).
BASE_RAGGED_CACHE_KEYS: frozenset[str] = frozenset({"prompt_embeds", "text_mask"})

PathExpect = Literal["file", "dir", "any"]


@dataclass(frozen=True)
class ComponentPath:
    """A ``[model]`` key naming a weights file/folder the pipeline loads.

    ``expect`` is what the loader accepts (``file``, ``dir`` or ``any``). ``env`` = the key can be
    overridden from ``RENGU_<env_prefix>_<KEY>`` (smoke/CI, ``config/local_env.py``). With a
    ``bundle_key`` on the spec, every other component is ``one_of(<component>, <bundle_key>)``:
    the bundle folder (diffusers layout) fills any component left empty.
    """

    key: str
    expect: PathExpect = "file"
    env: bool = True


@dataclass(frozen=True)
class QuantSpec:
    """Frozen-base quantization knobs a model honors (``model.transformer_fp8_matmul`` /
    ``model.transformer_4bit`` / ``model.fp8_grad_mode`` / ``model.fp8_matmul_dtype``).

    ``fp8``: ``"tensorwise"`` (always e4m3, no weight-format knob), ``"scaled_matmul"`` (weight
    format chosen by ``fp8_matmul_dtype`` from ``fp8_dtypes``, first = default) or None.
    ``grad_mode``: ``model.fp8_grad_mode`` (bf16/fp8 input-gradient GEMM) is exposed in the form.
    """

    fp8: Literal["tensorwise", "scaled_matmul"] | None = "tensorwise"
    nf4: bool = True
    grad_mode: bool = True
    fp8_dtypes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreviewSpec:
    """How a model renders previews and which ``[preview]`` defaults it gets.

    ``runner``: ``"sdxl"`` (diffusers pipeline + block-swap hooks) or ``"dit"`` (the model's own
    ``generate_preview_image``). ``steps`` / ``guidance`` are the sampler defaults a run uses when
    ``[preview]`` omits them; ``write_defaults`` = ``set_config_defaults`` writes them (and the
    other preview defaults) into the config, False = the runner falls back to them at sample time.
    ``memory_knobs``: the ``preview_<knob>`` memory options the runner honors
    (``offload_text_encoder``, ``blocks_to_swap``, ``offload_dit_for_decode``). ``edit``: renders
    prompts with ``control_images``. ``negative_prompt`` / ``defaults``: further ``[preview]``
    defaults (e.g. nanosaur2's ``guidance_mode``). ``help``: model-specific text of the sampler
    default fields, ``{"num_inference_steps" | "guidance_scale": {"description", "help", "doc"}}``
    (what the generic ``preview.*`` help cannot say, e.g. "samples without CFG").
    """

    runner: Literal["sdxl", "dit"]
    steps: int
    guidance: float
    write_defaults: bool = True
    memory_knobs: frozenset[str] = frozenset({"offload_text_encoder", "blocks_to_swap"})
    edit: bool = False
    negative_prompt: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    help: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterExport:
    """On-disk layout of an adapter export (``rengu_flow.networks.export_check`` checks it).

    ``file``: the safetensors name ``save_adapter`` writes. ``style``: ``"dotted"`` (PEFT-like keys
    ``<root><module.path>.<weight>``) or ``"kohya"`` (flat ``<root><module_path>.<weight>``, the
    module path with ``_`` for ``.``, weights ``lora_down.weight`` / ``lora_up.weight`` / ``alpha``
    for LoRA). ``roots``: the key prefixes every key starts with; the first one is required (the
    denoiser's), the others optional (e.g. text encoders). Empty = prefixes unchecked.
    """

    file: str = "adapter_model.safetensors"
    style: Literal["dotted", "kohya"] = "dotted"
    roots: tuple[str, ...] = ()


def _path_identity(path: Any) -> str:
    """``size:mtime_ns`` of a file, or a digest of a folder's (name, size, mtime_ns) entries.
    Empty when the path does not exist (tests, remote paths): nothing is added to the key."""
    import hashlib
    import os

    try:
        p = os.fspath(path)
        if os.path.isdir(p):
            entries = []
            for root, _dirs, files in os.walk(p):
                for name in files:
                    full = os.path.join(root, name)
                    st = os.stat(full)
                    entries.append((os.path.relpath(full, p), st.st_size, st.st_mtime_ns))
            return hashlib.sha1(repr(sorted(entries)).encode()).hexdigest()[:16]
        st = os.stat(p)
        return f"{st.st_size}:{st.st_mtime_ns}"
    except (OSError, TypeError, ValueError):
        return ""


@dataclass
class ModelCapability:
    """Descriptor of one canonical model type: training options, config fields and everything
    the integration surfaces (validation, defaults, env, installer, UI, preview) need about it."""

    type_id: str
    display_name: str
    adapters: list[str] = field(default_factory=list)
    full_finetune: bool = True
    # Derived from ``preview_spec`` when one is given (also sets features["preview"]).
    preview: bool = False
    aliases: list[str] = field(default_factory=list)
    branding_note: str = ""
    model_fields: list[dict[str, Any]] = field(default_factory=list)
    # Feature flags drive cross-section visibility (block_swap, preview, …) via field_visibility.
    features: dict[str, bool] = field(default_factory=dict)
    # Optional validation overrides — see model_config_rules.py (one_of, …).
    model_validation: dict[str, Any] = field(default_factory=dict)
    # Named adapter layer groups (adapter.layer_groups options). Names must match the
    # pipeline's ADAPTER_LAYER_GROUPS keys (guarded by tests/test_adapter_layer_groups.py).
    adapter_layer_groups: list[str] = field(default_factory=list)
    # The groups (a subset of ``adapter_layer_groups``) a default adapter trains when the config sets
    # neither ``layer_groups`` nor ``target_include``: a model whose reference scope is narrower than every
    # Linear of its target class. Empty = every targeted linear. The pipeline reads it from here; the UI
    # states it in the ``adapter.layer_groups`` placeholder (guarded by tests/contract).
    adapter_default_layer_groups: list[str] = field(default_factory=list)
    # Top-level module names of the adapter-targeted model, for torch-free preflight of
    # target_include/exclude typos (guarded against the real DiT by the same test file).
    adapter_module_roots: list[str] = field(default_factory=list)
    # Number of DiT transformer blocks (config-time [tread] route range check); 0 = unknown.
    transformer_blocks: int = 0
    # [model] keys that change what the text encoder produces, mapped to their default. Set
    # values (other than the default) join the text-embedding cache key, so changing one
    # re-encodes instead of silently reusing stale embeddings (see text_cache_identity()).
    text_cache_keys: dict[str, Any] = field(default_factory=dict)
    # Version of the cached-embedding layout/encoding. Bump it when the encoder's output for the
    # same caption changes (e.g. SDXL's pad-to-77): 0 (default) adds nothing, so untouched models
    # keep their caches; any other value joins the key and forces a re-encode.
    text_cache_version: int = 0
    # [model] keys naming the file(s) that hold the text encoder. The first one present joins the
    # cache key by NAME and size+mtime (never the path, so moving a dataset's run does not
    # invalidate it), so swapping the encoder/checkpoint re-encodes. A folder is fingerprinted by
    # its files' names, sizes and mtimes. A missing path adds nothing.
    text_cache_files: list[str] = field(default_factory=list)
    # [preview] keys restricted to fixed values for this model (validated by
    # model_config_rules.validate_preview_options), e.g. nanosaur2's guidance_mode.
    preview_options: dict[str, list[str]] = field(default_factory=dict)

    # ---- descriptor fields (model-spec refactor; consumers migrate in later phases) ----------
    # "<module>:<Class>" of the pipeline; the module is imported lazily by registry.models.
    pipeline: str = ""
    # RENGU_<env_prefix>_<KEY> env overrides of the component paths.
    env_prefix: str = ""
    # Weights paths the pipeline loads ([model] keys), in env/validation order.
    components: tuple[ComponentPath, ...] = ()
    # [model] key of a whole-bundle folder that fills any component left empty (one_of).
    bundle_key: str | None = None
    # model.cache_text_embeddings: "optional" (togglable, defaults to true) or "required" (must
    # stay true). None = undeclared (a pipeline registered only with @register_model): no default is
    # written and the UI toggle is hidden; the pipeline reads the raw config.
    text_cache: Literal["optional", "required"] | None = None
    quant: QuantSpec | None = None
    preview_spec: PreviewSpec | None = None
    # [model] defaults set_config_defaults applies for this model (beyond the generic ones).
    model_defaults: dict[str, Any] = field(default_factory=dict)
    # model.diffusion_model_dtype also becomes the default model.transformer_dtype (load dtype).
    diffusion_dtype_sets_load_dtype: bool = False
    # Install profiles (rengu_flow.install.profiles) a training run of this model needs.
    install_profiles: tuple[str, ...] = ()
    # User guide (repo-relative), default doc link of its fields' help.
    user_doc: str = ""
    # Key prefix of exported DiT adapters (pipeline ``adapter_export_prefix``, or
    # adapter_dit.save's default); None = the model exports with its own layout (sdxl).
    export_prefix: str | None = None
    # Per-adapter export layout, keyed by adapter type ("lycoris" = every lycoris_* kind); a kind
    # not listed exports ``adapter_model.safetensors`` with dotted keys under ``export_prefix``
    # (see adapter_export()).
    adapter_exports: dict[str, AdapterExport] = field(default_factory=dict)
    # Load dtypes this model cannot train in, {dtype name: why} (e.g. "float16"). Config validation and the
    # UI's dtype options derive from it; the pipeline refuses them too when built directly.
    refused_dtypes: dict[str, str] = field(default_factory=dict)
    # Module with the model's tiny-model test kit (contract tests); None = none yet.
    testkit: str | None = None
    # Cached tensors stored as ragged rows on top of BASE_RAGGED_CACHE_KEYS.
    ragged_cache_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # Features the UI schema gates on (``{"capability": ...}``), derived from the descriptor
        # so a model never lists them twice. Only true ones are stored.
        derived: dict[str, bool] = {}
        ps = self.preview_spec
        if ps is not None:
            self.preview = True
            derived["preview"] = True
            if ps.runner == "dit":
                derived["preview_dit_memory"] = True
            if "blocks_to_swap" in ps.memory_knobs:
                derived["preview_block_swap"] = True
            if "offload_dit_for_decode" in ps.memory_knobs:
                derived["preview_decode_offload"] = True
            if ps.edit:
                derived["edit"] = True
            # Model-specific [preview] options (fixed-value keys / own defaults) become
            # ``preview_<key>`` features (e.g. preview_guidance_mode).
            for key in (*self.preview_options, *ps.defaults):
                derived[f"preview_{key}"] = True
        if self.text_cache == "optional":
            derived["text_cache_optional"] = True
        if self.adapter_layer_groups:
            derived["adapter_layer_groups"] = True
        # "preview" stays the first feature key (to_dict order is the API's).
        explicit = {k: v for k, v in self.features.items() if k not in derived}
        self.features = {**derived, **explicit}

    @property
    def pipeline_module(self) -> str:
        """Module part of ``pipeline`` ("" when unset)."""
        return self.pipeline.partition(":")[0]

    @property
    def pipeline_class_name(self) -> str:
        """Class part of ``pipeline`` ("" when unset)."""
        return self.pipeline.partition(":")[2]

    def adapter_export(self, adapter_type: str) -> AdapterExport:
        """The export layout of ``adapter_type`` (``adapter_exports``, else the DiT default)."""
        kind = str(adapter_type).lower()
        declared = self.adapter_exports.get(kind)
        if declared is None and kind.startswith("lycoris_"):
            declared = self.adapter_exports.get("lycoris")
        if declared is not None:
            return declared
        return AdapterExport(roots=(self.export_prefix,) if self.export_prefix else ())

    def text_cache_identity(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """The text-encoder settings of ``model_config`` that key the text-embedding cache.
        Empty when nothing differs from the defaults, which keeps existing caches valid."""
        identity = {
            key: str(model_config[key])
            for key, default in self.text_cache_keys.items()
            if model_config.get(key) is not None and model_config[key] != default
        }
        if self.text_cache_version:
            identity["text_cache_version"] = str(self.text_cache_version)
        for key in self.text_cache_files:
            if model_config.get(key):
                file_id = _path_identity(model_config[key])
                if file_id:
                    identity["text_encoder_file"] = f"{key}:{file_id}"
                break
        return identity

    def training_modes(self) -> list[str]:
        modes: list[str] = []
        if self.full_finetune:
            modes.append("full_finetune")
        modes.extend(self.adapters)
        return modes

    def to_dict(self) -> dict[str, Any]:
        # Exact whitelist: the web API (GET /schema registries.model_capabilities) is unchanged by
        # the descriptor fields above.
        resolved_labels: dict[str, str] = {
            kind: DEFAULT_ADAPTER_LABELS.get(kind, kind)
            for kind in self.adapters
        }
        return {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "adapters": list(self.adapters),
            "adapter_labels": resolved_labels,
            "full_finetune": self.full_finetune,
            "preview": self.preview,
            "aliases": list(self.aliases),
            "branding_note": self.branding_note,
            "training_modes": self.training_modes(),
            "model_fields": list(self.model_fields),
            "features": dict(self.features),
            "model_validation": dict(self.model_validation),
            "adapter_layer_groups": list(self.adapter_layer_groups),
            "adapter_module_roots": list(self.adapter_module_roots),
        }

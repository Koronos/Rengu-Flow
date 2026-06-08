"""Entry point for Rengu Flow. Load config, validate; run training loop when not dry-run."""

import argparse
import functools
import glob
import os
import shutil
from pathlib import Path

from rengu_flow.config import (
    apply_local_config_to_environ,
    load_config,
    load_dataset_config,
    load_eval_dataset_config,
    load_local_config,
    set_config_defaults,
    validate_config,
)
from rengu_flow.config.validation import ConfigValidationError
from rengu_flow.control.progress_stream import ProgressEmitter


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Rengu Flow: TOML-driven training (Phase 1).")
    parser.add_argument(
        "--config",
        required=False,
        help="Path to TOML configuration file (not required with --dump_dataset).",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank from distributed launcher.")
    parser.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=None)
    parser.add_argument(
        "--run_dir",
        default=None,
        help=(
            "Pin the run folder (name under output_dir, or absolute path) without resuming a "
            "checkpoint. Lets 'continue from scratch' reuse a run's folder and train from step 0, "
            "keeping one folder per run."
        ),
    )
    parser.add_argument("--reset_dataloader", action="store_true")
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.add_argument("--reset_optimizer_params", action="store_true")
    parser.add_argument("--regenerate_cache", action="store_true")
    parser.add_argument(
        "--regenerate_text_cache",
        action="store_true",
        help="Rebuild metadata and text embeddings only; reuse latents when possible.",
    )
    parser.add_argument("--cache_only", action="store_true", help="Cache then exit (no-op in Phase 1 minimal).")
    parser.add_argument("--trust_cache", action="store_true")
    parser.add_argument("--i_know_what_i_am_doing", action="store_true")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--dump_dataset", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="Load config, apply defaults, validate, then exit (no dataset load, no training).")
    try:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    return parser.parse_args(argv)


def _distributed_init(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = args.local_rank
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
    # Keep MASTER_PORT from deepspeed.launcher when already set.
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    return world_size, rank, local_rank


def _get_most_recent_run_dir(output_dir: str) -> str:
    """Return the most recently modified run directory under output_dir."""
    dirs = [d for d in glob.glob(os.path.join(output_dir, "*")) if os.path.isdir(d)]
    if not dirs:
        raise ValueError(f"No run directories found under {output_dir}")
    return max(dirs, key=os.path.getmtime)


def _build_optimizer(
    model_parameters,
    *,
    config,
    model,
    pipeline_model,
    ds_config,
    global_batch_size,
    gradient_release,
):
    """Construct the pipeline optimizer (lifted verbatim from ``_run_training``).

    Handles the empty-param dummy, gradient-release per-parameter optimizers,
    and GenericOptim / weight-decay param-group splitting. Imports stay lazy so
    importing ``main`` does not require torch/deepspeed.
    """
    import inspect

    import torch
    import deepspeed

    from rengu_flow.optim import resolve_optimizer_class
    from rengu_flow.utils import is_main_process
    from rengu_flow.optim.param_groups import (
        adjust_beta2_half_life,
        split_genericoptim_param_groups,
        split_weight_decay_param_groups,
    )
    from rengu_flow.vendor.diffusion_pipe_optimizers.gradient_release import (
        GradientReleaseOptimizerWrapper,
    )

    if len(model_parameters) == 0:
        from collections import defaultdict
        class DummyOptimizer(torch.optim.Optimizer):
            def __init__(self):
                self.state = defaultdict(dict)
                self.param_groups = []
            def step(self, closure=None): pass
            def zero_grad(self, set_to_none=True): pass
        return DummyOptimizer()

    optim_cfg_raw = config["optimizer"]
    optim_type = optim_cfg_raw["type"]
    optim_type_lower = optim_type.lower()
    optim_config = adjust_beta2_half_life(
        {k: v for k, v in optim_cfg_raw.items() if k not in ("type", "gradient_release")},
        global_batch_size,
    )
    if "beta2_half_life" in optim_cfg_raw and is_main_process():
        print(f"Computed beta2 = {optim_config['betas'][1]}")

    opt_args: list = []
    kwargs = dict(optim_config)
    klass = resolve_optimizer_class(optim_type)

    if optim_type_lower == "offload":
        opt_args.append(torch.optim.AdamW)
        kwargs["fused"] = True

    if gradient_release:
        def _report_progress(self, step):
            lr = self.get_lr()
            mom = self.get_mom()
            deepspeed.utils.logging.log_dist(
                f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}",
                ranks=[0],
            )
        deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

        def _exec_reduce_grads(self):
            assert self.mpu.get_data_parallel_world_size() == 1, (
                "When using gradient_release, data parallel world size must be 1. "
                "Use pipeline_stages = num_gpus."
            )
            return

        deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[
            deepspeed.runtime.pipe.schedule.ReduceGrads
        ] = _exec_reduce_grads

        def add_(self, *args, **kwargs):
            self.data.add_(*args, **kwargs)

        for p in model_parameters:
            p.add_ = add_.__get__(p)

        if "foreach" in inspect.signature(klass).parameters:
            kwargs["foreach"] = False

        gas = ds_config["gradient_accumulation_steps"]
        if "betas" in kwargs:
            kwargs["betas"] = [b ** (1 / gas) for b in kwargs["betas"]]
        if "momentum" in kwargs:
            kwargs["momentum"] = kwargs["momentum"] ** (1 / gas)

        optimizer_dict = {}
        for pg in model.get_param_groups(model_parameters):
            param_kwargs = kwargs.copy()
            if isinstance(pg, dict):
                for p in pg["params"]:
                    param_kwargs["lr"] = pg.get("lr", param_kwargs.get("lr"))
                    optimizer_dict[p] = klass([p], **param_kwargs)
            else:
                optimizer_dict[pg] = klass([pg], **param_kwargs)

        def optimizer_hook(p):
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()

        # Register only on params that actually got a per-parameter optimizer. get_param_groups may
        # drop/freeze some (e.g. Cosmos lr=0 groups → requires_grad_(False)); those are not in
        # optimizer_dict, and torch refuses a post-accumulate hook on a tensor without a gradient.
        for p in optimizer_dict:
            p.register_post_accumulate_grad_hook(optimizer_hook)

        return GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))

    if optim_type_lower == "genericoptim":
        kwargs["compile"] = config.get("compile", False)
        kwargs["mpu"] = pipeline_model.mpu()
        param_groups = split_genericoptim_param_groups(
            model.get_param_groups(model_parameters),
            kwargs,
        )
    else:
        param_groups = model.get_param_groups(model_parameters)

    param_groups = split_weight_decay_param_groups(param_groups, optim_type_lower)
    return klass(param_groups, *opt_args, **kwargs)


def _maybe_start_profiler():
    """Env-gated torch profiler. RENGU_PROF_DIR enables it; tune the window with
    RENGU_PROF_WAIT/WARMUP/ACTIVE (defaults skip compile warmup, capture 6 steady steps).
    on_trace_ready writes a kernel table (sorted by CUDA time) + a chrome trace, then it stops."""
    import os

    prof_dir = os.environ.get("RENGU_PROF_DIR")
    if not prof_dir:
        return None
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    os.makedirs(prof_dir, exist_ok=True)
    wait = int(os.environ.get("RENGU_PROF_WAIT", "12"))
    warmup = int(os.environ.get("RENGU_PROF_WARMUP", "3"))
    active = int(os.environ.get("RENGU_PROF_ACTIVE", "6"))

    def _on_ready(p):
        tbl = p.key_averages().table(sort_by="cuda_time_total", row_limit=45)
        with open(os.path.join(prof_dir, "kernels_cuda.txt"), "w") as fh:
            fh.write(tbl)
        tbl2 = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=45)
        with open(os.path.join(prof_dir, "kernels_self_cuda.txt"), "w") as fh:
            fh.write(tbl2)
        try:
            p.export_chrome_trace(os.path.join(prof_dir, "trace.json"))
        except Exception as e:  # trace export is best-effort
            print(f"[prof] chrome trace export failed: {e}", flush=True)
        print(f"[prof] wrote profiler tables + trace to {prof_dir}", flush=True)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=_on_ready,
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    )
    prof.start()
    print(f"[prof] torch profiler armed (wait={wait} warmup={warmup} active={active}) -> {prof_dir}", flush=True)
    return prof


def _setup_compile_disk_cache(config, is_main):
    """Configure the TorchInductor/Triton disk cache for torch.compile.

    Returns (enabled, cache_dir). Sets the TORCHINDUCTOR_* / TRITON_CACHE_DIR env
    vars (in every rank, before compile) when caching should be active. Honors
    config["compile_disk_cache"] ("auto"/true/false) and config["compile_cache_dir"].
    See MEASURED FACTS: caching only helps with static shapes, and the dir must be
    on an ext4 (255-char filename) filesystem.
    """
    raw = config.get("compile_disk_cache", "auto")
    mode = str(raw).lower() if isinstance(raw, str) else raw
    compile_dynamic = config.get("compile_dynamic") is True

    if mode is True or mode == "true":
        if compile_dynamic and is_main:
            print(
                "[compile-cache] compile_disk_cache=true but compile_dynamic is on; "
                "the disk cache likely won't help (shape guards won't match).",
                flush=True,
            )
    elif mode == "auto":
        if compile_dynamic:
            return False, None
    else:
        # false or any unrecognized value -> never cache.
        return False, None

    # Default to a subdir of the dataset cache root (cache_root) — it's the install's folder for
    # regenerable cache artifacts (same nature as the compile cache), is on the same disk the user
    # already chose for caches, and follows a custom cache_root. Falls back to <repo>/cache/compile.
    cache_dir = config.get("compile_cache_dir")
    if not cache_dir:
        try:
            from rengu_flow.data.cache_paths import resolve_cache_root
            cache_dir = str(resolve_cache_root(config) / "compile")
        except Exception:
            cache_dir = str(Path(__file__).resolve().parent.parent / "cache" / "compile")
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    triton_dir = os.path.join(cache_dir, "triton")

    try:
        os.makedirs(triton_dir, exist_ok=True)
    except OSError as exc:
        if is_main:
            print(f"[compile-cache] disabled: could not create cache dir {cache_dir} ({exc}).", flush=True)
        return False, None

    # Safety check: filesystem must accept a ~200-char filename (Triton kernel-cache
    # filenames are ~150 chars; eCryptfs home dirs cap at ~143 and raise ENAMETOOLONG).
    probe = os.path.join(cache_dir, "c" * 200)
    try:
        with open(probe, "w"):
            pass
        os.remove(probe)
    except OSError:
        if is_main:
            print(
                f"[compile-cache] disabled: cache dir {cache_dir} is on a filesystem with a "
                "short filename limit (compile cache needs ext4/255-char; e.g. an encrypted "
                "home). Set compile_cache_dir to an ext4 path to enable.",
                flush=True,
            )
        return False, None

    # Respect existing env (user override) but still enable the FX/autograd caches.
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)
    os.environ.setdefault("TRITON_CACHE_DIR", triton_dir)
    return True, cache_dir


def _run_training(args, config):
    import torch
    import deepspeed
    from deepspeed import comm as dist
    from torch.utils.tensorboard import SummaryWriter

    from rengu_flow.utils.common import empty_cuda_cache
    import time

    from rengu_flow.utils.bench import bench_enabled, bench_init, bench_record, bench_summarize
    from rengu_flow.utils.training_metrics import log_training_step
    import rengu_flow.utils.common as common_module

    model_dtype = config["model"].get("dtype")
    forward_dtype = config["model"].get("diffusion_model_dtype") or model_dtype
    if hasattr(torch, "float16") and isinstance(forward_dtype, torch.dtype):
        common_module.AUTOCAST_DTYPE = forward_dtype

    from rengu_flow.registry import get_model
    from rengu_flow.optim import apply_warmup, resolve_scheduler
    from rengu_flow.utils import ManualPipelineModule, get_data_iterator_for_step, is_main_process
    from rengu_flow.utils.saver import Saver
    from rengu_flow.utils.eval import evaluate
    from rengu_flow.utils.gen_probe import generalization_probe
    from rengu_flow.data import (
        Dataset,
        DatasetManager,
        PipelineDataLoader,
        SyntheticSDXLDataset,
        validate_dataset_config_for_real_data,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    world_size, rank, local_rank = _distributed_init(args)
    deepspeed.init_distributed()
    if torch.cuda.is_available():
        device_rank = local_rank if local_rank >= 0 else (dist.get_rank() if dist is not None else 0)
        torch.cuda.set_device(device_rank)
    if os.environ.get("RENGU_TUNING_TF32_APPLY") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = get_model(config)
    model.load_diffusion_model()

    train_data = None
    eval_data_map = {}
    dataset_config = config.get("_dataset_config_loaded")
    use_real_dataset = (
        dataset_config is not None
        and config.get("synthetic_num_batches") is None
    )
    if use_real_dataset:
        validate_dataset_config_for_real_data(dataset_config)
        train_data = Dataset(
            dataset_config,
            model,
            skip_dataset_validation=args.i_know_what_i_am_doing,
        )
        dataset_manager = DatasetManager(
            model,
            regenerate_cache=args.regenerate_cache or args.regenerate_text_cache,
            trust_cache=args.trust_cache,
            caching_batch_size=config.get("caching_batch_size", 1),
            cache_num_proc=config.get("cache_num_proc"),
            cache_keep_in_memory=config.get("cache_keep_in_memory", False),
            cache_format=config.get("cache_format", "v2"),
            cache_dedup_text_embeddings=config.get("cache_dedup_text_embeddings", False),
        )
        dataset_manager.register(train_data)
        for i, eval_entry in enumerate(config.get("eval_datasets", [])):
            name, eval_dataset_config = load_eval_dataset_config(eval_entry)
            eval_data = Dataset(
                eval_dataset_config,
                model,
                skip_dataset_validation=args.i_know_what_i_am_doing,
            )
            eval_data_map[name] = eval_data
            dataset_manager.register(eval_data)
        dataset_manager.cache()
        if args.cache_only:
            if is_main_process():
                print("rengu_flow: cache complete (--cache_only). Exiting.")
            return

    is_adapter = bool(config.get("adapter"))
    if adapter_config := config.get("adapter"):
        model.configure_adapter(adapter_config)
        if adapter_config.get("init_from_existing"):
            model.load_adapter_weights(adapter_config["init_from_existing"])
    else:
        if config["model"].get("freeze_text_encoders", False):
            model.freeze_text_encoders()

    # Block swapping is supported for adapters and for full-model training. Full-model swap
    # requires gradient_release: each block's optimizer step then runs inside the backward while
    # that block is resident on the GPU. A monolithic optimizer.step() would instead need every
    # trainable block resident at once, defeating the swap. DeepSpeed places the model on the GPU
    # normally; prepare_block_swap_training() (after initialize) pushes swappable blocks to CPU,
    # and the offloader pulls them back on demand.
    if config.get("blocks_to_swap", 0):
        if config.get("pipeline_stages", 1) != 1:
            raise ValueError("Block swapping requires pipeline_stages = 1.")
        if not config.get("adapter") and not config["optimizer"].get("gradient_release"):
            raise ValueError(
                "Block swapping for full-model training requires optimizer.gradient_release = true "
                "(the per-parameter optimizer step must run during the backward pass while each "
                "block is on the GPU)."
            )
        # Keep DeepSpeed from hauling the whole model onto the GPU at init / broadcasting CPU blocks;
        # the offloader + prepare_block_swap_training own placement (see block_swap docstring).
        from rengu_flow.training.block_swap import patch_deepspeed_for_block_swap

        patch_deepspeed_for_block_swap()
        model.enable_block_swap(config["blocks_to_swap"])

    layers = model.to_layers()
    num_stages = config.get("pipeline_stages", 1)
    partition_method = config.get("partition_method", "parameters")
    partition_split = config.get("partition_split")
    if partition_split is None and num_stages > 1:
        partition_split = [len(layers) // num_stages] * (num_stages - 1)
    elif partition_split is None:
        partition_split = []

    activation_checkpointing = config.get("activation_checkpointing", False)
    extra_kw = {}
    if activation_checkpointing:
        # interval N = checkpoint every N transformer blocks (1 = every block, most memory-saving /
        # most recompute). Raise to recompute less at the cost of more activation VRAM.
        extra_kw["activation_checkpoint_interval"] = int(config.get("activation_checkpoint_interval", 1))
        extra_kw["checkpointable_layers"] = model.checkpointable_layers
        if activation_checkpointing == "unsloth":
            from rengu_flow.utils.unsloth_utils import unsloth_checkpoint
            extra_kw["activation_checkpoint_func"] = unsloth_checkpoint
        elif activation_checkpointing == "selective":
            # Selective Activation Checkpointing (SAC): SAVE the expensive attention outputs
            # (don't recompute them in backward), recompute only the cheaper ops. Middle ground
            # between full recompute (lowest VRAM, ~33% recompute tax) and no-checkpoint (OOM at
            # high res). Saves the attention-recompute tax while keeping VRAM in budget.
            from functools import partial

            from torch.utils.checkpoint import (
                CheckpointPolicy,
                checkpoint,
                create_selective_checkpoint_contexts,
            )

            _aten = torch.ops.aten
            _save_ops = {
                _aten._scaled_dot_product_flash_attention.default,
                _aten._scaled_dot_product_efficient_attention.default,
                _aten._scaled_dot_product_cudnn_attention.default,
            }
            # extra ops to also save (e.g. "mm,bmm,addmm") via config, for tuning the VRAM/speed point
            for _name in str(config.get("selective_checkpoint_save_ops", "")).replace(" ", "").split(","):
                if _name and hasattr(_aten, _name):
                    _save_ops.add(getattr(_aten, _name).default)

            def _sac_policy(ctx, op, *a, **kw):
                return CheckpointPolicy.MUST_SAVE if op in _save_ops else CheckpointPolicy.PREFER_RECOMPUTE

            def _selective_checkpoint(function, *args, **kwargs):
                return checkpoint(
                    function, *args, use_reentrant=False,
                    context_fn=partial(create_selective_checkpoint_contexts, _sac_policy), **kwargs,
                )

            extra_kw["activation_checkpoint_func"] = _selective_checkpoint

            # SAC uses MORE VRAM than full checkpointing (it keeps the saved activations instead of
            # recomputing them). It is opt-in, but warn loudly — and extra-loudly on small GPUs —
            # so a low-VRAM user who enabled it isn't surprised by an OOM. Tunable / revertible:
            #   - revert to safe full checkpointing: activation_checkpointing = true
            #   - dial the VRAM/speed tradeoff: selective_checkpoint_save_ops (fewer ops = less VRAM)
            from rengu_flow.utils import is_main_process as _imp
            if _imp():
                _gb = 0.0
                try:
                    _gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                except Exception:
                    pass
                _saved = sorted(o._name if hasattr(o, "_name") else str(o) for o in _save_ops)
                _msg = (
                    f"[checkpoint] activation_checkpointing='selective' (SAC): saving {len(_save_ops)} op type(s) "
                    f"{_saved}, recomputing the rest. This is faster than full checkpointing but uses MORE VRAM. "
                    "If you OOM: set activation_checkpointing=true (full, safe) or shrink "
                    "selective_checkpoint_save_ops."
                )
                if 0 < _gb < 12.0:
                    _msg += (f" WARNING: this GPU has ~{_gb:.0f} GB — SAC may not fit; full checkpointing "
                             "(activation_checkpointing=true) is the safer choice here.")
                print(_msg, flush=True)
        else:
            from functools import partial
            extra_kw["activation_checkpoint_func"] = partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=config.get("reentrant_activation_checkpointing", False),
            )

    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split if partition_method == "manual" else None,
        loss_fn=model.get_loss_fn(),
        **extra_kw,
    )
    if config.get("compile"):
        compile_kwargs = {}
        if compile_mode := config.get("compile_mode"):
            compile_kwargs["mode"] = compile_mode
        compile_dynamic = config.get("compile_dynamic") is True
        if compile_dynamic:
            compile_kwargs["dynamic"] = True
        cache_enabled, cache_dir = _setup_compile_disk_cache(config, is_main_process())

        # ====================================================================
        # REJECTED — DO NOT MERGE TO main. See BRANCH_REJECTED_DO_NOT_MERGE.md.
        # Measured net-negative for production (Cosmos LoKr, 4080, multi-res):
        # naive per-block compile halves the one-time cold-compile spike
        # (77s->38s) but costs +2-5%/step FOREVER (graph breaks); crossover
        # ~1000-1600 steps, so real long runs come out slower overall. The
        # "proper" fix (nested_compile_region) is a dud here too: crashes on
        # dynamic shapes (torch 2.12) and gives ~no compile win on static
        # (compile is inductor-bound, not trace-bound). Kept opt-in default-off
        # for short/debug runs only. Do not enable by default; do not merge.
        # ====================================================================
        # Regional (per-block) compile: compile each identical transformer block on its
        # own so inductor compiles ONE artifact and reuses it across all ~28 blocks (and
        # per shape), shrinking the cold-compile / per-shape recompile spike to ~one
        # block's cost instead of the whole-pipeline graph. Unsafe with block swapping
        # (blocks stream CPU<->GPU), so fall back to whole-model compile in that case.
        regional = bool(config.get("compile_regional"))
        if regional and config.get("blocks_to_swap", 0) > 0:
            if is_main_process():
                print(
                    "[compile] compile_regional requested but blocks_to_swap > 0; the "
                    "swapped blocks stream CPU<->GPU and are unsafe to compile in place. "
                    "Falling back to whole-model compile.",
                    flush=True,
                )
            regional = False

        if regional:
            # Compile the blocks in place. The TransformerLayer wrappers built by
            # to_layers() (and fed to ManualPipelineModule) cache `self.block`, so update
            # both the ModuleList and the wrappers' references to the compiled blocks.
            from rengu_flow.model.cosmos_predict2.layers import TransformerLayer as _TL

            blocks = model.transformer.blocks
            num_blocks = len(blocks)
            compiled = {}
            for i in range(num_blocks):
                cb = torch.compile(blocks[i], **compile_kwargs)
                compiled[id(blocks[i])] = cb
                blocks[i] = cb
            # Re-point the already-constructed TransformerLayer wrappers at the compiled
            # blocks (they were created from the uncompiled ones in to_layers()).
            for layer in layers:
                if isinstance(layer, _TL):
                    new_block = compiled.get(id(layer.block))
                    if new_block is not None:
                        layer.block = new_block
            if is_main_process():
                print(
                    f"torch.compile (regional) per block x{num_blocks} "
                    f"({compile_kwargs or 'defaults'})",
                    flush=True,
                )
                msg = (
                    f"[compile] torch.compile enabled REGIONAL (mode={compile_mode or 'default'}, "
                    f"dynamic={compile_dynamic}): compiling 1 transformer block, reused across "
                    f"{num_blocks} - much smaller compile/recompile spikes. The FIRST training "
                    "step per resolution/shape still compiles kernels (~one block's cost) - this "
                    "is NORMAL, not a hang."
                )
                if cache_enabled:
                    msg += (
                        f" Disk cache: {cache_dir} (subsequent runs with the same static "
                        "shapes skip recompile)."
                    )
                if compile_dynamic:
                    msg += (
                        " (dynamic shapes: each new resolution/AR-bucket recompiles; disk "
                        "cache does not help dynamic.)"
                    )
                print(msg, flush=True)
        else:
            if is_main_process():
                print(f"pipeline_model.compile({compile_kwargs or 'defaults'})", flush=True)
                msg = (
                    f"[compile] torch.compile enabled (mode={compile_mode or 'default'}, "
                    f"dynamic={compile_dynamic}). The FIRST training step per resolution/shape "
                    "compiles kernels and may take ~1-4 min - this is NORMAL, not a hang."
                )
                if cache_enabled:
                    msg += (
                        f" Disk cache: {cache_dir} (subsequent runs with the same static "
                        "shapes skip recompile)."
                    )
                if compile_dynamic:
                    msg += (
                        " (dynamic shapes: each new resolution/AR-bucket recompiles; disk "
                        "cache does not help dynamic.)"
                    )
                print(msg, flush=True)
            pipeline_model.compile(**compile_kwargs)
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    micro_batch = config.get("micro_batch_size_per_gpu", 1)
    if isinstance(micro_batch, dict):
        micro_batch = list(micro_batch.values())[0]
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    world_size_for_opt = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size_for_opt = micro_batch * gradient_accumulation_steps * world_size_for_opt

    gradient_release = config["optimizer"].get("gradient_release", False)
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 0.0 if gradient_release else config.get("gradient_clipping", 1.0),
        "steps_per_print": config.get("steps_per_print", 1),
    }

    get_optimizer = functools.partial(
        _build_optimizer,
        config=config,
        model=model,
        pipeline_model=pipeline_model,
        ds_config=ds_config,
        global_batch_size=global_batch_size_for_opt,
        gradient_release=gradient_release,
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )
    model_engine._support_torch_style_backward = True
    model_engine._configure_optimizer(get_optimizer, parameters_to_train)
    optimizer = model_engine.optimizer

    # DeepSpeed has now placed the model on the GPU; push swappable blocks back to CPU so steady
    # state stays under the VRAM ceiling. The offloader's hooks pull each block on demand.
    if config.get("blocks_to_swap", 0):
        model.prepare_block_swap_training()

    from rengu_flow.training.ema import TrainingEMA

    training_ema = TrainingEMA.from_config(config, parameters_to_train)
    if hasattr(model_engine, "communication_data_type"):
        model_engine.communication_data_type = (
            config["adapter"]["dtype"] if config.get("adapter") else config["model"]["dtype"]
        )

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    micro_batch_dict = config.get("micro_batch_size_per_gpu", 1)
    if not isinstance(micro_batch_dict, dict):
        micro_batch_dict = {None: micro_batch_dict}
    image_micro_batch_dict = config.get("image_micro_batch_size_per_gpu", micro_batch_dict)
    if not isinstance(image_micro_batch_dict, dict):
        image_micro_batch_dict = {None: image_micro_batch_dict}

    if train_data is not None:
        train_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            micro_batch_dict,
            gradient_accumulation_steps,
            image_micro_batch_dict,
        )
        for eval_data in eval_data_map.values():
            eval_data.post_init(
                model_engine.grid.get_data_parallel_rank(),
                model_engine.grid.get_data_parallel_world_size(),
                micro_batch_dict,
                gradient_accumulation_steps,
                image_micro_batch_dict,
            )
    else:
        num_batches = config.get("synthetic_num_batches", 50)
        train_data = SyntheticSDXLDataset(
            num_batches=num_batches,
            micro_batch_size=micro_batch,
            latent_channels=4,
            latent_height=64,
            latent_width=64,
        )

    _loader_kwargs = dict(
        num_dataloader_workers=config.get("dataloader_num_workers", 0),
        dataloader_prefetch=config.get("dataloader_prefetch", False),
        pin_memory=config.get("dataloader_pin_memory", False),
        prefetch_factor=config.get("dataloader_prefetch_factor", 2),
        persistent_workers=config.get("dataloader_persistent_workers", True),
    )
    train_dataloader = PipelineDataLoader(
        train_data,
        model_engine,
        gradient_accumulation_steps,
        model,
        **_loader_kwargs,
    )
    eval_gradient_accumulation_steps = config.get("eval_gradient_accumulation_steps", 1)
    eval_dataloaders = {
        name: PipelineDataLoader(
            eval_data,
            model_engine,
            eval_gradient_accumulation_steps,
            model,
            **_loader_kwargs,
        )
        for name, eval_data in eval_data_map.items()
    }
    # Deterministic generalization probe (val loss / train probe / GAP). The val curve comes
    # from the first explicit eval dataset (held-out by construction); the train probe is a
    # cheap separate loader over the train dataset (forward-only, capped to a fixed number of
    # batches — never advances the training iterator). Both no-op when unavailable.
    val_gap_enable = config.get("val_gap_enable", True)
    val_gap_probe_batches = config.get("val_gap_probe_batches", 8)
    val_probe_dataloader = None
    train_probe_dataloader = None
    if val_gap_enable and eval_dataloaders:
        # Use the first eval dataset as the held-out validation source for the gap.
        val_probe_dataloader = next(iter(eval_dataloaders.values()))
        if train_data is not None:
            train_probe_dataloader = PipelineDataLoader(
                train_data,
                model_engine,
                eval_gradient_accumulation_steps,
                model,
                **_loader_kwargs,
            )
    elif val_gap_enable and is_main_process():
        print(
            "rengu_flow: val_gap_enable is set but no eval_datasets are configured; "
            "the train-val gap probe is disabled (add an eval_datasets entry to enable it)."
        )
    # A resolution schedule samples only a subset of resolutions per epoch, so the
    # live dataloader length shrinks; measure steps_per_epoch over the full set of
    # resolutions (full_epoch_len) to keep total_steps/progress stable.
    schedule_active = getattr(train_data, "schedule_active", False)
    if schedule_active:
        steps_per_epoch = max(
            1, train_data.full_epoch_len // gradient_accumulation_steps
        )
    else:
        steps_per_epoch = max(1, len(train_dataloader) // gradient_accumulation_steps)
    epochs = config.get("epochs", 1)
    total_steps = epochs * steps_per_epoch
    # The schedule is measured against a step budget: the user's max_steps if set
    # (that option wins), otherwise the system-derived total_steps. Stage boundaries
    # and the LR horizon follow this same target so they line up with the real run.
    max_steps = config.get("max_steps")
    schedule_target_steps = max_steps if max_steps is not None else total_steps
    if schedule_active:
        train_data.set_schedule_target(schedule_target_steps)
    lr_horizon_steps = schedule_target_steps if schedule_active else total_steps
    lr_scheduler = resolve_scheduler(
        config.get("lr_scheduler", "constant"),
        optimizer,
        config,
        lr_horizon_steps,
        steps_per_epoch,
    )
    lr_scheduler = apply_warmup(optimizer, lr_scheduler, config.get("warmup_steps", 0))
    model_engine.lr_scheduler = lr_scheduler

    # Establish run_dir on rank 0 and broadcast (compatible with diffusion-pipe multi-GPU)
    output_dir = config["output_dir"]
    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None else config.get("resume_from_checkpoint", False)
    )
    run_dir_container = [None]
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        if args.run_dir:
            # Folder pinned by the caller (e.g. 'continue from scratch'): reuse this exact run
            # folder regardless of whether we resume a checkpoint. Decoupled from
            # resume_from_checkpoint so one run keeps one folder.
            run_dir_container[0] = (
                args.run_dir
                if os.path.isabs(args.run_dir)
                else os.path.join(output_dir, args.run_dir)
            )
        elif resume_from_checkpoint is True:
            run_dir_container[0] = _get_most_recent_run_dir(output_dir)
        elif isinstance(resume_from_checkpoint, str):
            if os.path.isabs(resume_from_checkpoint) and os.path.isdir(resume_from_checkpoint):
                run_dir_container[0] = resume_from_checkpoint
            else:
                run_dir_container[0] = os.path.join(output_dir, resume_from_checkpoint)
            if not os.path.exists(run_dir_container[0]):
                raise ValueError(f"Checkpoint directory {run_dir_container[0]} does not exist")
        else:
            from rengu_flow.run_naming import build_run_folder_name

            folder_name = build_run_folder_name(config.get("run_name"))
            run_dir_container[0] = os.path.join(output_dir, folder_name)
    if dist is not None and hasattr(torch.distributed, "broadcast_object_list"):
        torch.distributed.broadcast_object_list(
            run_dir_container, src=0, group=dist.get_world_group()
        )
    run_dir = run_dir_container[0]
    if run_dir is None:
        raise RuntimeError("run_dir was not set on rank 0")
    os.makedirs(run_dir, exist_ok=True)
    # When the folder is pinned (--run_dir) and a specific checkpoint tag was requested
    # (e.g. global_step40), resume that tag from inside the folder; otherwise DeepSpeed
    # resumes the folder's `latest`. (Without --run_dir, --resume_from_checkpoint names a
    # run folder, handled above.)
    resume_tag = None
    if args.run_dir and isinstance(resume_from_checkpoint, str) and resume_from_checkpoint:
        if os.path.isdir(os.path.join(run_dir, resume_from_checkpoint)):
            resume_tag = resume_from_checkpoint
    # Sweep any signal files left over from a prior run (e.g. a save_quit from a force-stop that
    # killed the process before a step consumed it) so this run doesn't quit on its first step.
    from rengu_flow.utils.signal_files import clear_stale_signals

    if is_main_process():
        for stale in clear_stale_signals(run_dir):
            print(f"Cleared stale signal file: {stale}")
    if is_main_process() and not resume_from_checkpoint:
        shutil.copy(args.config, run_dir)
        if config.get("dataset") and os.path.isfile(config["dataset"]):
            shutil.copy(config["dataset"], run_dir)
    dist.barrier()

    if is_main_process():
        print(f"Training: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
        print(f"Run dir: {run_dir}")
        print(
            f"TensorBoard: tensorboard --logdir {output_dir} "
            f"(pick run {os.path.basename(run_dir)} in the sidebar; do not point logdir at the run folder itself)"
        )

    global_batch_size = micro_batch * gradient_accumulation_steps
    if hasattr(dist, "get_world_size"):
        global_batch_size *= dist.get_world_size()
    if config.get("eval_every_n_examples") is not None:
        config["eval_every_n_steps"] = config["eval_every_n_examples"] // global_batch_size
        if is_main_process():
            print(f"Computed eval_every_n_steps = {config['eval_every_n_steps']}")
    if config.get("save_every_n_examples") is not None:
        config["save_every_n_steps"] = config["save_every_n_examples"] // global_batch_size
        if is_main_process():
            print(f"Computed save_every_n_steps = {config['save_every_n_steps']}")
    saver = Saver(
        args,
        config,
        is_adapter,
        run_dir,
        model,
        train_dataloader,
        model_engine,
        pipeline_model,
        steps_per_epoch=steps_per_epoch,
    )
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    wandb_enable = config.get("monitoring", {}).get("enable_wandb", False)
    if wandb_enable and is_main_process():
        try:
            import wandb
            mon = config.get("monitoring", {})
            if mon.get("wandb_api_key"):
                wandb.login(key=mon["wandb_api_key"])
            wandb.init(
                project=mon.get("wandb_tracker_name", "rengu-flow"),
                name=mon.get("wandb_run_name", run_dir),
                config=config,
                dir=run_dir,
            )
        except ImportError:
            wandb_enable = False
    disable_block_swap_for_eval = config.get("disable_block_swap_for_eval", False)
    x_axis_examples = config.get("x_axis_examples", False)
    eval_every_n_steps = config.get("eval_every_n_steps")
    eval_every_n_epochs = config.get("eval_every_n_epochs")
    eval_before_first_step = config.get("eval_before_first_step", True)
    from rengu_flow.utils.preview import get_preview_config, previews_configured, reload_preview_config, run_previews, should_run_previews

    preview_before_first_step = get_preview_config(config).get("preview_before_first_step", False)
    disable_block_swap_for_preview = config.get(
        "disable_block_swap_for_preview", disable_block_swap_for_eval
    )

    step = 1
    examples = global_batch_size
    epoch = train_dataloader.epoch
    # When a resolution schedule is active, the run is governed by the schedule's
    # step budget (max_steps if set, else the epochs-derived total) rather than by
    # the epoch count, since staged epochs are shorter than a full-resolution epoch.
    effective_max_steps = schedule_target_steps if schedule_active else max_steps
    logging_steps = config.get("logging_steps", 1)
    from rengu_flow.training_progress import (
        TrainingProgressTracker,
        budget_display_epoch,
        build_progress_payload,
    )

    progress_tracker = TrainingProgressTracker(
        max_steps=max_steps,
        total_steps=total_steps,
        loss_window=steps_per_epoch,
    )
    # Throttled stdout progress markers (rank 0). Replaces per-step log spam and
    # per-iteration status.json writes; the web UI parses these for its live bar.
    progress_emitter = ProgressEmitter() if is_main_process() else None
    final_model_name = None
    checkpointed = False
    saved = False
    last_save_step = -1  # last step at which an inference model was saved (for the final save)
    epoch_loss = 0.0
    num_steps = 0
    # Latest generalization-probe result (val loss / train probe / gap), surfaced in the live
    # progress marker so the UI can show it next to the train loss. None until the first probe.
    last_val_metrics: dict[str, float] | None = None
    bench_csv = (
        bench_init(run_dir)
        if bench_enabled(config) and is_main_process()
        else None
    )
    per_step_batch = micro_batch * gradient_accumulation_steps

    if resume_from_checkpoint:
        load_lr = "force_constant_lr" not in config and not args.reset_optimizer and not args.reset_optimizer_params
        load_optimizer = not args.reset_optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if param_groups is not None:
            param_groups = param_groups.copy()
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            tag=resume_tag,
            load_module_strict=False,
            load_lr_scheduler_states=load_lr,
            load_optimizer_states=load_optimizer,
        )
        if args.reset_optimizer_params and param_groups is not None:
            optimizer.param_groups = param_groups
        dist.barrier()
        if load_path is None:
            if is_main_process():
                print("Resume requested but no checkpoint found; starting from step 1.")
        else:
            if args.reset_dataloader:
                train_dataloader.epoch = client_state["custom_loader"]["epoch"]
            else:
                train_dataloader.load_state_dict(client_state["custom_loader"])
            step = client_state["step"] + 1
            examples = client_state.get("examples", (step - 1) * global_batch_size) + global_batch_size
            epoch = train_dataloader.epoch
            if is_main_process():
                print(f"Resuming from checkpoint at epoch {epoch}, step {step}")

    if eval_before_first_step and not resume_from_checkpoint and eval_dataloaders:
        empty_cuda_cache()
        if hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()
        evaluate(
            model,
            model_engine,
            eval_dataloaders,
            tb_writer,
            0,
            eval_gradient_accumulation_steps,
            disable_block_swap_for_eval,
            optimizer=optimizer,
            wandb_enable=wandb_enable,
        )
        if val_probe_dataloader is not None:
            last_val_metrics = generalization_probe(
                model,
                model_engine,
                val_probe_dataloader,
                train_probe_dataloader,
                tb_writer,
                0,
                eval_gradient_accumulation_steps,
                disable_block_swap_for_eval,
                probe_batches=val_gap_probe_batches,
                optimizer=optimizer,
                wandb_enable=wandb_enable,
            )

    if (
        preview_before_first_step
        and not resume_from_checkpoint
        and previews_configured(config)
    ):
        run_previews(
            model,
            config,
            tb_writer,
            0,
            disable_block_swap=disable_block_swap_for_preview,
            optimizer=optimizer,
            wandb_enable=wandb_enable,
        )

    oom_skip_cfg = config.get("train", {}).get("oom_skip", {})
    oom_skip_enabled = bool(oom_skip_cfg.get("enabled", False))
    # Emergency checkpoint on an otherwise-fatal CUDA OOM (e.g. during a preview/eval) so the run is
    # resumable instead of lost. Best-effort: a failed save never masks the original error.
    save_on_oom = bool(config.get("train", {}).get("save_checkpoint_on_oom", True))
    from rengu_flow.utils.oom_skip import is_cuda_oom

    oom_skip_state = None
    if oom_skip_enabled:
        from rengu_flow.utils.oom_skip import OomSkipState, handle_oom_skip

        oom_skip_state = OomSkipState(max_consecutive=int(oom_skip_cfg.get("max_consecutive", 3)))

    train_seed = int(config.get("train_seed", 42))
    import random as _random

    import numpy as np

    _random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    if train_data is not None and hasattr(train_data, "set_training_context"):
        train_data.set_training_context(train_seed, step)
    # Seed the schedule's notion of the current step (handles resume too) so the
    # first epoch rollover selects the correct stage. Count epochs by the budget-relative
    # epoch (1..epochs) too — dataloader epochs are short (one resolution per stage), so
    # using them would overshoot `epochs` for saves/eval/naming.
    if schedule_active:
        train_data.current_step = step
        epoch = budget_display_epoch(step, steps_per_epoch, epochs)

    # Optional env-gated torch profiler (RENGU_PROF_DIR=/path). Skips compile/warmup steps, captures
    # a steady window, and on completion writes a key_averages table + chrome trace. Zero cost off.
    _prof = _maybe_start_profiler() if is_main_process() else None

    try:
        while True:
            model_engine.reset_activation_shape()
            if train_data is not None and hasattr(train_data, "set_training_context"):
                train_data.set_training_context(train_seed, step)
            # Step-accurate resolution schedule: switch the active resolution(s) the
            # moment this step crosses a stage boundary, restarting iteration mid-epoch
            # if needed. Must run before pulling this step's micro-batches.
            if schedule_active:
                train_dataloader.refresh_for_step(step)
            iterator = get_data_iterator_for_step(train_dataloader, model_engine)
            t0 = time.perf_counter()
            skipped_oom = False
            if oom_skip_enabled:
                try:
                    loss = model_engine.train_batch(iterator).item()
                except Exception as e:
                    if not is_cuda_oom(e):
                        raise
                    handle_oom_skip(
                        oom_skip_state,
                        model_engine,
                        clear_cache=bool(oom_skip_cfg.get("clear_cache_on_skip", True)),
                        step=step,
                        tb_writer=tb_writer,
                    )
                    oom_skip_state.record_skip()
                    train_dataloader.sync_epoch()
                    skipped_oom = True
            else:
                loss = model_engine.train_batch(iterator).item()
            iter_sec = time.perf_counter() - t0
            if _prof is not None:
                _prof.step()
            if skipped_oom:
                if effective_max_steps is not None and step >= effective_max_steps:
                    final_model_name = f"step{step}"
                    if is_main_process():
                        print(f"Reached max_steps={effective_max_steps}")
                    break
                if not schedule_active and epoch > epochs:
                    final_model_name = f"epoch{epoch}"
                    if is_main_process():
                        print(f"Reached epochs={epochs}")
                    break
                step += 1
                examples += global_batch_size
                continue
            if oom_skip_state is not None:
                oom_skip_state.record_success()
            if training_ema is not None:
                training_ema.update(parameters_to_train)
            if bench_enabled(config) and is_main_process():
                bench_record(
                    bench_csv,
                    step=step,
                    loss=loss,
                    iter_sec=iter_sec,
                    batch_size=per_step_batch,
                )
            train_dataloader.sync_epoch()
            epoch_loss += loss
            num_steps += 1

            saver.set_status_context(step, examples, epoch, loss)
            if schedule_active:
                # Count/name epochs by the budget-relative epoch (full epochs), not the
                # short single-resolution dataloader epochs.
                budget_epoch = budget_display_epoch(step, steps_per_epoch, epochs)
                new_epoch, checkpointed_ep, saved_ep = saver.process_epoch(
                    epoch,
                    step,
                    examples,
                    effective_epoch=budget_epoch,
                    advanced=budget_epoch != epoch,
                )
            else:
                new_epoch, checkpointed_ep, saved_ep = saver.process_epoch(epoch, step, examples)
            if checkpointed_ep:
                checkpointed = True
            if saved_ep:
                saved = True
                last_save_step = step
            finished_epoch = new_epoch is not None and new_epoch != epoch
            if new_epoch is None:
                final_model_name = f"epoch{epoch}"
                break
            if new_epoch != epoch:
                epoch = new_epoch

            x_axis = examples if x_axis_examples else step
            progress_tracker.record_step_duration(iter_sec)
            progress_tracker.record_loss(loss)
            step_progress = progress_tracker.metrics(step=step)
            # Throttled progress marker to stdout (rank 0). Always emit on the final step
            # and on save/epoch boundaries so the UI never misses a transition; otherwise
            # the emitter caps the rate (~1/sec). No per-step log line, no status.json.
            # With a resolution schedule, dataloader epochs are short (one stage = a
            # subset of resolutions), so the raw epoch counter overshoots the configured
            # `epochs`. Report a budget-relative epoch (1..epochs) instead — total steps
            # are unchanged, only the displayed epoch is made meaningful again.
            display_epoch = (
                budget_display_epoch(step, steps_per_epoch, epochs)
                if schedule_active
                else epoch
            )
            if progress_emitter is not None:
                is_final = effective_max_steps is not None and step >= effective_max_steps
                progress_emitter.emit(
                    build_progress_payload(
                        step=step,
                        loss=loss,
                        epoch=display_epoch,
                        metrics=step_progress,
                        val_metrics=last_val_metrics,
                    ),
                    force=is_final or finished_epoch or saved,
                )
            log_training_step(
                tb_writer=tb_writer,
                wandb_enable=wandb_enable,
                optimizer=optimizer,
                loss=loss,
                x_axis=x_axis,
                step=step,
                logging_steps=logging_steps,
                is_main=is_main_process(),
            )

            if (eval_every_n_steps and step % eval_every_n_steps == 0) or (
                finished_epoch and eval_every_n_epochs and epoch % eval_every_n_epochs == 0
            ):
                evaluate(
                    model,
                    model_engine,
                    eval_dataloaders,
                    tb_writer,
                    x_axis,
                    eval_gradient_accumulation_steps,
                    disable_block_swap_for_eval,
                    optimizer=optimizer,
                    wandb_enable=wandb_enable,
                )
                if val_probe_dataloader is not None:
                    last_val_metrics = generalization_probe(
                        model,
                        model_engine,
                        val_probe_dataloader,
                        train_probe_dataloader,
                        tb_writer,
                        x_axis,
                        eval_gradient_accumulation_steps,
                        disable_block_swap_for_eval,
                        probe_batches=val_gap_probe_batches,
                        optimizer=optimizer,
                        wandb_enable=wandb_enable,
                    )
                    # Re-emit a fresh marker so the UI surfaces the new val/gap promptly
                    # (the per-step emit above ran before this probe).
                    if progress_emitter is not None and last_val_metrics is not None:
                        progress_emitter.emit(
                            build_progress_payload(
                                step=step,
                                loss=loss,
                                epoch=display_epoch,
                                metrics=step_progress,
                                val_metrics=last_val_metrics,
                            ),
                            force=True,
                        )

            if finished_epoch:
                if is_main_process() and num_steps > 0:
                    avg_epoch_loss = epoch_loss / num_steps
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
                    if wandb_enable:
                        try:
                            import wandb
                            wandb.log({"train/epoch_loss": avg_epoch_loss, "epoch": epoch})
                        except ImportError:
                            pass
                epoch_loss = 0.0
                num_steps = 0

            step_checkpointed, step_saved, step_signals = saver.process_step(step, examples)
            if step_checkpointed:
                checkpointed = True
            if step_saved:
                saved = True
                last_save_step = step

            # Hot-reload the [preview] section from the config file on a `reload_config`
            # signal (edit the TOML, then signal). Applies live to the checks below; only
            # [preview] is reloaded — model/optimizer/dataset can't change mid-run.
            if step_signals.should_reload_config:
                reload_preview_config(config, args.config)

            preview_x_axis = examples if x_axis_examples else step
            if should_run_previews(
                config,
                step,
                epoch,
                finished_epoch=finished_epoch,
                forced=step_signals.should_preview,
            ):
                run_previews(
                    model,
                    config,
                    tb_writer,
                    preview_x_axis,
                    disable_block_swap=disable_block_swap_for_preview,
                    optimizer=optimizer,
                    wandb_enable=wandb_enable,
                )

            if effective_max_steps is not None and step >= effective_max_steps:
                final_model_name = f"step{step}"
                if is_main_process():
                    print(f"Reached max_steps={effective_max_steps}")
                break
            if not schedule_active and epoch > epochs:
                final_model_name = f"epoch{epoch}"
                if is_main_process():
                    print(f"Reached epochs={epochs}")
                break
            step += 1
            examples += global_batch_size
    except Exception as exc:
        # Last-ditch checkpoint on a fatal CUDA OOM (commonly during a preview/eval) so the run can
        # resume instead of being lost. Best-effort and rank-0 gated logging; the original error is
        # always re-raised. Note: in multi-GPU runs the save's collective may not complete if only
        # some ranks OOM — the re-raise still tears the job down.
        if save_on_oom and is_cuda_oom(exc) and not checkpointed:
            if is_main_process():
                print(
                    f"rengu_flow: CUDA OOM at step {step} — attempting emergency checkpoint "
                    "before exit...",
                    flush=True,
                )
            try:
                empty_cuda_cache()
                if saver.save_checkpoint(step, examples):
                    checkpointed = True
                    if is_main_process():
                        print(
                            f"rengu_flow: emergency checkpoint saved at step {step}; resume with "
                            "--resume_from_checkpoint.",
                            flush=True,
                        )
            except Exception as save_exc:  # noqa: BLE001 - never mask the original OOM
                if is_main_process():
                    print(
                        f"rengu_flow: emergency checkpoint FAILED after OOM: {save_exc}",
                        flush=True,
                    )
        raise

    if not checkpointed:
        if saver.save_checkpoint(step, examples):
            checkpointed = True
    # Always produce the final model unless the most recent save was already at this exact
    # step. A periodic save (e.g. save_every_n_epochs) that fired earlier must not suppress
    # the final weights — that left a run ending on max_steps mid-epoch without a final model.
    if final_model_name and last_save_step != step:
        saver.save_model(final_model_name)
    saver.shutdown_async_exports()

    if is_main_process():
        if bench_enabled(config):
            bench_summarize(
                bench_csv,
                config.get("run_name", "bench"),
                run_dir,
            )
        print("Training complete.")


def run_prepared(args) -> None:
    if args.dump_dataset is not None:
        from rengu_flow.data.dump_dataset import dump_dataset

        dump_dataset(args.dump_dataset)
        return
    if args.config is None:
        raise SystemExit("rengu_flow: --config is required (unless using --dump_dataset).")

    config = load_config(args.config)
    set_config_defaults(config)

    if not args.validate_only:
        dataset_config = load_dataset_config(config)
        if dataset_config is not None:
            config["_dataset_config_loaded"] = dataset_config

    try:
        validate_config(config, for_script=True)
    except ConfigValidationError as e:
        raise SystemExit(f"Config validation failed: {e}") from e

    if args.validate_only:
        return

    if args.cache_only and not config.get("_dataset_config_loaded"):
        print("rengu_flow: --cache_only requires a dataset config; exiting.")
        return

    _run_training(args, config)


def main(argv: list[str] | None = None):
    load_local_config()
    apply_local_config_to_environ()
    # Re-apply after [training.env] may have set expandable_segments (rengu_flow/__init__ already
    # ran it pre-torch-import; this catches a value injected by apply_local_config_to_environ).
    from rengu_flow.platform_compat import configure_cuda_allocator

    configure_cuda_allocator()
    run_prepared(parse_args(argv))


if __name__ == "__main__":
    main()

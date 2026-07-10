"""Entry point for Rengu Flow. Load config, validate; run training loop when not dry-run."""

import argparse
import contextlib
import functools
import glob
import os
import shutil
from pathlib import Path

from rengu_flow.config import (
    apply_local_config_to_environ,
    apply_model_paths_from_env,
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
    parser = argparse.ArgumentParser(description="Rengu Flow: TOML-driven training.")
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
    parser.add_argument("--cache_only", action="store_true", help="Cache then exit without training.")
    parser.add_argument("--trust_cache", action="store_true")
    parser.add_argument("--i_know_what_i_am_doing", action="store_true")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--dump_dataset", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true", help="Load config, apply defaults, validate, then exit (no dataset load, no training).")
    # ``deepspeed.add_config_arguments`` only registers the args the DeepSpeed launcher injects at
    # train time (``--deepspeed`` etc., none of which this code reads) — but importing DeepSpeed for
    # it costs ~17s. The lightweight modes (``--validate-only`` / ``--dump_dataset``) are run
    # directly, never via the launcher, so they never see those args: skip the import for them.
    import sys

    argv_to_scan = sys.argv[1:] if argv is None else argv
    if not any(flag in argv_to_scan for flag in ("--validate-only", "--dump_dataset")):
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
    os.environ.setdefault("MASTER_ADDR", "localhost")
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

    from rengu_flow.registry.optimizers import get_optimizer_class
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
    klass = get_optimizer_class(optim_type)

    if optim_type_lower == "offload":
        opt_args.append(torch.optim.AdamW)
        kwargs["fused"] = True

    if gradient_release:
        import deepspeed  # gradient_release patches the DeepSpeed engine (deepspeed backend only)

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
    import sys
    import torch

    from rengu_flow import distributed as dist
    from rengu_flow.engine import select_backend
    from rengu_flow.utils.common import empty_cuda_cache
    import time

    # Engine backend: 'deepspeed' (pipeline, multi-GPU, default Linux) or 'accelerate'
    # (single-GPU plain torch, default Windows — no DeepSpeed import at all). DeepSpeed is
    # imported only inside the deepspeed branches below so native Windows needs no DeepSpeed.
    backend_obj = select_backend(config)
    backend_obj.validate(config)  # raises ValueError for unsupported feature combos

    from rengu_flow.utils.bench import bench_enabled, bench_init, bench_record, bench_summarize
    from rengu_flow.utils.training_metrics import install_grad_norm_capture, log_training_step
    import rengu_flow.utils.common as common_module

    # Startup banner: rengu-flow version + the dep versions that actually move per-step speed and
    # numerics. Logged once on rank 0 so a saved log is self-describing (which version produced these
    # numbers? which torch?) — invaluable when debugging a perf/behaviour change across updates.
    from rengu_flow.utils import is_main_process as _is_main
    from rengu_flow.utils.logging import tag_third_party_console_logs
    from rengu_flow.version import installed_version as _iv, version_string as _vs

    # Third-party logging (lycoris, datasets, transformers, ...) stays in the log file
    # (auditable) but tagged [dep:<pkg>] so the trainer's own narrative stays readable.
    tag_third_party_console_logs()
    if _is_main():
        import sys as _sys
        _deps = " ".join(f"{_n}={_iv(_n) or '?'}" for _n in ("torch", "kaon", "deepspeed", "triton"))
        print(
            f"[rengu-flow] version {_vs()} | python {_sys.version.split()[0]} | {_deps}",
            flush=True,
        )

    model_dtype = config["model"].get("dtype")
    forward_dtype = config["model"].get("diffusion_model_dtype") or model_dtype
    if hasattr(torch, "float16") and isinstance(forward_dtype, torch.dtype):
        common_module.AUTOCAST_DTYPE = forward_dtype

    from rengu_flow.registry import get_model
    from rengu_flow.optim import apply_warmup, resolve_scheduler
    from rengu_flow.utils import get_data_iterator_for_step, is_main_process
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

    from rengu_flow.platform_compat import PLATFORM

    if PLATFORM.torch_file_system_sharing:
        # file_system sharing strategy is POSIX-only (shared-memory files); raises on Windows.
        torch.multiprocessing.set_sharing_strategy("file_system")
    world_size, rank, local_rank = _distributed_init(args)
    if backend_obj.is_distributed:
        import deepspeed

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
            training_config=config,
        )
        dataset_manager = DatasetManager(
            model,
            regenerate_cache=args.regenerate_cache or args.regenerate_text_cache,
            trust_cache=args.trust_cache,
            caching_batch_size=config.get("caching_batch_size", 1),
            cache_num_proc=config.get("cache_num_proc"),
            cache_keep_in_memory=config.get("cache_keep_in_memory", False),
            backend=backend_obj,
        )
        dataset_manager.register(train_data)
        for i, eval_entry in enumerate(config.get("eval_datasets", [])):
            name, eval_dataset_config = load_eval_dataset_config(eval_entry)
            eval_data = Dataset(
                eval_dataset_config,
                model,
                skip_dataset_validation=args.i_know_what_i_am_doing,
                training_config=config,
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
    # Capability/config validation (unsupported backend, non-adapter accelerate, pipeline_stages>1)
    # is handled by backend_obj.validate() called above; only the setup side-effect remains here.
    if config.get("blocks_to_swap", 0) and backend_obj.is_distributed:
        # Keep DeepSpeed from hauling the whole model onto the GPU at init / broadcasting CPU
        # blocks; the offloader + prepare_block_swap_training own placement. The accelerate engine
        # has no such init move to neutralize (TorchEngine skips its .to(device) when swapping).
        from rengu_flow.training.block_swap import patch_deepspeed_for_block_swap
        patch_deepspeed_for_block_swap()
        model.enable_block_swap(config["blocks_to_swap"])
    elif config.get("blocks_to_swap", 0):
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
    if activation_checkpointing == "auto":
        # Compiler-driven AC: no manual checkpoint wrapper at all — Inductor's
        # memory-budget partitioner picks the save/recompute split per compiled
        # layer graph (see training/activation_budget.py).
        from rengu_flow.training.activation_budget import (
            apply_activation_memory_budget,
            resolve_auto_ac_budget,
        )

        _budget = resolve_auto_ac_budget(config)
        apply_activation_memory_budget(_budget)
        if is_main_process():
            print(
                f"[checkpoint] activation_checkpointing='auto': compile's memory-budget "
                f"partitioner with activation_memory_budget={_budget} (exact recompute "
                "chosen per graph; 0.0 ~ full-checkpoint VRAM, 1.0 ~ no-checkpoint speed).",
                flush=True,
            )
    elif activation_checkpointing:
        # interval N = checkpoint every N transformer blocks (1 = every block, most memory-saving /
        # most recompute). Raise to recompute less at the cost of more activation VRAM.
        # ('selective'/'unsloth' were retired — defaults.py degrades them to true with a warning;
        # see docs/EXPERIMENTS_GRAVEYARD.md and activation_checkpointing='auto' for the fast path.)
        from functools import partial

        extra_kw["activation_checkpoint_interval"] = int(config.get("activation_checkpoint_interval", 1))
        extra_kw["checkpointable_layers"] = model.checkpointable_layers
        extra_kw["activation_checkpoint_func"] = partial(
            torch.utils.checkpoint.checkpoint,
            use_reentrant=config.get("reentrant_activation_checkpointing", False),
        )

    pipeline_model = backend_obj.build_pipe(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split if partition_method == "manual" else None,
        loss_fn=model.get_loss_fn(),
        extra_kw=extra_kw,
    )
    if config.get("compile"):
        from rengu_flow.training.compile_plan import apply_dynamo_limits, plan_compile

        num_shapes = None
        if train_data is not None:
            shapes = set(train_data.distinct_size_buckets())
            for eval_data in eval_data_map.values():
                shapes |= eval_data.distinct_size_buckets()
            num_shapes = len(shapes)
        compile_plan = plan_compile(config, num_shapes)
        apply_dynamo_limits(compile_plan)
        compile_dynamic = config.get("compile_dynamic") is True
        cache_enabled, cache_dir = _setup_compile_disk_cache(config, is_main_process())
        if is_main_process():
            print(f"pipeline_model.compile({compile_plan.kwargs or 'defaults'})", flush=True)
            if compile_dynamic:
                # One graph per distinct micro-batch size: PyTorch always
                # specializes size-1 dims, so e.g. {512 = 2, 1024 = 1} costs two
                # cold compiles; resolutions/AR buckets share those graphs.
                _mb_cfg = config.get("micro_batch_size_per_gpu", 1)
                _signatures = sorted(
                    set(_mb_cfg.values()) if isinstance(_mb_cfg, dict) else {_mb_cfg}
                )
                msg = (
                    f"[compile] torch.compile enabled (mode={config.get('compile_mode') or 'default'}, "
                    f"dynamic=True). Expect {len(_signatures)} cold compile(s) of ~1-4 min "
                    f"(one per distinct micro-batch size: {_signatures}); after that, shapes "
                    "share the graph(s) and only the occasional unusual geometry recompiles "
                    "once. Disk cache does not apply to dynamic graphs."
                )
            else:
                msg = (
                    f"[compile] torch.compile enabled (mode={config.get('compile_mode') or 'default'}, "
                    f"dynamic=False). The FIRST training step per resolution/shape "
                    "compiles kernels and may take ~1-4 min - this is NORMAL, not a hang."
                )
                if cache_enabled:
                    msg += (
                        f" Disk cache: {cache_dir} (subsequent runs with the same static "
                        "shapes skip recompile)."
                    )
            print(msg, flush=True)
            for note in compile_plan.notes:
                print(f"[compile] {note}", flush=True)
        if config.get("compile_scope", "model") == "block":
            # Compile each transformer block alone, leaving the layer orchestration
            # (activation checkpointing, block-swap hooks, tuple plumbing) eager. The
            # whole-module trace graph-breaks on all of those and measured ~0% on a
            # swapped base; block scope keeps one clean fusable graph per block
            # (1.14x bf16, 1.78x with fp8 tensorwise — krea2 4080 microbench 2026-07).
            class _SeqDynamicShim(torch.nn.Module):
                """Pin the sequence dim as dynamic on every call: under non-reentrant AC
                the recompute otherwise can select a different compiled graph than the
                forward (static->dynamic transition; pytorch#166926) and checkpointing
                aborts with a CheckpointError."""

                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, hidden, *args, **kwargs):
                    torch._dynamo.mark_dynamic(hidden, 1)
                    return self.inner(hidden, *args, **kwargs)

            reentrant_ac = bool(config.get("reentrant_activation_checkpointing", False))
            if not reentrant_ac:
                # pytorch#166926 workarounds: the AC-recompute checkpoint metadata
                # comparison only exists on the NON-reentrant path. Reentrant AC skips
                # the comparison entirely, and disabling LRU reordering there is pure
                # harm — FIFO eviction thrashes the shared block cache into repeated
                # cold recompiles at every train<->eval guard flip.
                try:
                    torch._C._dynamo.eval_frame._set_lru_cache(False)
                except AttributeError:
                    pass
            n_blocks = 0
            for m in pipeline_model.modules():
                inner = getattr(m, "block", None)
                if isinstance(inner, torch.nn.Module):
                    compiled = torch.compile(inner, **compile_plan.kwargs)
                    m.block = (
                        _SeqDynamicShim(compiled)
                        if compile_dynamic and not reentrant_ac
                        else compiled
                    )
                    n_blocks += 1
            if is_main_process():
                print(f"[compile] block scope: compiled {n_blocks} transformer blocks", flush=True)
        else:
            pipeline_model.compile(**compile_plan.kwargs)
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    # With a per-resolution dict, a single representative number is still needed
    # here (DeepSpeed config, optimizer batch scaling): the mean of the values —
    # example/step accounting below switches to the dataset's real per-step
    # average once post_init has sized every bucket.
    from rengu_flow.training.activation_budget import nominal_micro_batch

    micro_batch = nominal_micro_batch(config.get("micro_batch_size_per_gpu", 1))
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    world_size_for_opt = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size_for_opt = micro_batch * gradient_accumulation_steps * world_size_for_opt

    gradient_release = config["optimizer"].get("gradient_release", False)
    if gradient_release and not backend_obj.supports_gradient_release:
        raise ValueError(
            f"optimizer.gradient_release requires engine='deepspeed' (it patches the DeepSpeed "
            f"pipeline engine); engine={backend_obj.name!r} does not support it yet."
        )
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

    model_engine = backend_obj.build_engine(
        pipeline_model=pipeline_model,
        ds_config=ds_config,
        args=args,
        get_optimizer=get_optimizer,
        parameters_to_train=parameters_to_train,
    )
    optimizer = model_engine.optimizer

    # DeepSpeed discards the grad norm it computes while clipping the bf16/fp32 optimizer path, so
    # train/grad_norm never logs for AdamW/Prodigy. Re-wrap the clip to keep that value.
    install_grad_norm_capture(model_engine)

    # DeepSpeed has now placed the model on the GPU; push swappable blocks back to CPU so steady
    # state stays under the VRAM ceiling. The offloader's hooks pull each block on demand.
    if config.get("blocks_to_swap", 0):
        model.prepare_block_swap_training()

    # Saved-activation offload to pinned CPU RAM (None when disabled). Wraps each
    # train_batch below; previews/eval run under no_grad and save nothing.
    from rengu_flow.training.activation_offload import ActivationOffloader

    act_offloader = ActivationOffloader.from_config(
        config, params_provider=pipeline_model.parameters
    )
    if act_offloader is not None and is_main_process():
        print(
            "[act-offload] activation_offload enabled: saved activations stream to "
            "pinned CPU RAM over side streams (per-step volume logged after the "
            "first step).",
            flush=True,
        )

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
        dataloader_prefetch=config.get("dataloader_prefetch", True),
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
    # Per-shape compile announces are only true in STATIC mode (each shape
    # really compiles there). Under compile_dynamic one graph per micro-batch
    # signature serves every shape — announcing each data shape as "compiling"
    # was measured to be pure noise (most new shapes run in ~1-2 s).
    # The activation budget is GLOBAL in both modes (applied at startup): the
    # retired per-shape scaling is in docs/EXPERIMENTS_GRAVEYARD.md — it baked
    # the first shape's budget into the single dynamic graph (OOM at any
    # configured base) and needed batch-aware token math to not OOM static.
    train_dataloader.announce_new_shapes = (
        bool(config.get("compile"))
        and config.get("compile_dynamic") is not True
        and is_main_process()
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
    # total_steps is the authority; an "epoch" is just an N/epochs slice for cadence
    # (save/eval/preview/display), not a clean pass over the data. One nominal epoch is
    # the schedule-weighted optimizer-step count: each bucket contributes len(bucket)*phi,
    # where phi is the fraction of the run its resolution is active (phi=1 with no
    # schedule, so this matches the old full-resolution count). len(bucket) already folds
    # in the per-resolution batch and gradient accumulation, so the budget shrinks exactly
    # by the resolutions a schedule drops -- no extra division, and the stage fractions
    # stay exact on the [0, N] step axis regardless of where epoch boundaries fall.
    schedule_active = getattr(train_data, "schedule_active", False)
    steps_per_epoch = max(1, round(train_data.scheduled_epoch_len()))
    # Caption variants (multi-line .txt) multiply the iteration order, but they are
    # regularization samples of the same images, not new data. Divide them out so an
    # "epoch" still means one pass over the images — save/eval/preview cadence and
    # the epochs*steps_per_epoch budget stay stable for any K, and the variants
    # rotate across epochs (each pass serves the next per-image variant).
    caption_variants = max(1, int(getattr(train_data, "caption_variants", 1)))
    if caption_variants > 1:
        steps_per_epoch = max(1, steps_per_epoch // caption_variants)
        if is_main_process():
            print(
                f"[data] {caption_variants} caption variants per image: epoch "
                f"accounting uses {steps_per_epoch} steps/epoch (one pass over the "
                "images); variants rotate across epochs.",
                flush=True,
            )
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

    # WSD: the step where the LR leaves the constant phase and the decay tail begins. The loop
    # saves a protected pre-decay "fork" checkpoint here so the run can be extended from before
    # the decay (resume from it with a larger length; the tail re-anchors to the new end).
    wsd_fork_step = None
    if config.get("lr_scheduler", "constant").lower() == "wsd":
        from rengu_flow.optim.resolver import wsd_decay_onset_step

        onset = wsd_decay_onset_step(config, lr_horizon_steps)
        if onset > 0:
            wsd_fork_step = onset
            if is_main_process():
                print(
                    f"rengu_flow: WSD — LR held constant until step {onset}, then a decay tail; a "
                    f"protected 'predecay' fork checkpoint is saved at step {onset} (extend the run "
                    "by resuming from it with a larger epochs/max_steps)."
                )

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
    if dist.is_initialized():
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
    if isinstance(config.get("micro_batch_size_per_gpu"), dict) and hasattr(
        train_data, "avg_examples_per_step"
    ):
        # Per-resolution batches: the per-step batch varies by bucket, so the
        # example accounting (examples counter, eval/save_every_n_examples)
        # uses the dataset's real weighted average instead of the nominal mean.
        global_batch_size = max(1, round(train_data.avg_examples_per_step()))
        if is_main_process():
            print(
                f"Per-resolution micro batch: example accounting uses the real "
                f"average of {global_batch_size} examples/step."
            )
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
    # Centralized tracking client: one sink fans out to the configured backends (manifest /
    # tensorboard / optional wandb). Built only on rank 0; other ranks get a no-op NullSink so
    # the loop stays identical without per-call rank guards. Connect/disconnect = [tracking].enabled.
    from rengu_track import (
        EVENT_FAILED,
        EVENT_FINISHED,
        EVENT_RESTARTED_FROM_SCRATCH,
        EVENT_RESUMED,
        EVENT_RUN_STARTED,
        EVENT_STOP_REQUESTED,
        NullSink,
        build_sink,
        read_events,
    )

    sink = build_sink(config, run_dir) if is_main_process() else NullSink()
    tracking_cfg = config.get("tracking", {})
    if is_main_process() and tracking_cfg.get("enabled", True) and tracking_cfg.get(
        "capture_lineage", True
    ):
        from rengu_track import lineage

        sink.set_lineage(lineage.capture())
        sink.set_hardware(lineage.hardware())
    disable_block_swap_for_eval = config.get("disable_block_swap_for_eval", False)
    # Block swap allocates/frees a fresh GPU copy of each swapped block every step. That churn,
    # interleaved with variable-size activation allocations, slowly grows the caching allocator's
    # *reserved* pool (fragmentation) — ~0.3 GB per 50 steps observed on krea2 — until an otherwise
    # 13 GB run OOMs. eval/preview already call empty_cuda_cache() and reset it, which is why runs
    # *with* previews never showed the creep; a run with neither reclaims nothing. Reclaim on a
    # sparse cadence so pure-training swap runs stay bounded (a sync every 50 steps costs nothing
    # next to the swap traffic). Gated on block swap so non-swap bench baselines are untouched.
    block_swap_active = bool(config.get("blocks_to_swap", 0))
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

    # Background system-metrics sampler (rank 0): pushes system/* scalars (GPU util/VRAM/temp/
    # power, CPU/RAM) at a fixed interval, tagged with the live `step`. Daemon thread; stopped on
    # both exit paths so the peak/mean aggregates flush to the manifest.
    sampler = None
    sampler_cfg = tracking_cfg.get("system_sampler", {})
    if is_main_process() and tracking_cfg.get("enabled", True) and sampler_cfg.get("enabled", True):
        from rengu_track.sampler import SystemSampler

        sampler = SystemSampler(
            sink,
            interval_sec=sampler_cfg.get("interval_sec", 10),
            step_fn=lambda: step,
        )
        sampler.start()

    from rengu_flow.training_progress import (
        EpochSchedule,
        TrainingProgressTracker,
        budget_reached_target,
        build_progress_payload,
        plan_final_saves,
    )

    # Single epoch authority for the whole loop — naming, save/eval cadence, progress, and
    # termination all read epoch numbers from here, derived from the step. The run is a fixed
    # step budget (max_steps if set, else epochs*steps_per_epoch) split into `epochs` equal
    # step chunks — identical with or without a resolution schedule, so components can no longer
    # disagree. The dataloader keeps its own internal epoch only for shuffling/seeding.
    epoch_schedule = EpochSchedule(steps_per_epoch, epochs)
    total_budget_steps = schedule_target_steps  # = max_steps if set, else total_steps
    # Duck-typed progress consumers (TREAD route layers); collected once, pushed per step.
    route_progress_layers = [
        m for m in pipeline_model.modules() if hasattr(m, "set_training_progress")
    ]
    epoch = epoch_schedule.current(step)
    logging_steps = config.get("logging_steps", 1)

    def _log_resolution_exposure() -> None:
        """Estimate how many times each image is trained per resolution and log it (rank 0).

        With a resolution schedule, resolutions that appear in fewer stages are trained fewer
        times; this surfaces that imbalance up front so epochs/stages can be sized to a target.
        Best-effort: a reporting failure must never break training.
        """
        if not is_main_process():
            return
        try:
            from rengu_flow.data.exposure import (
                estimate_image_exposure,
                format_exposure_report,
                schedule_stage_spans,
            )

            weight: dict[int, float] = {}
            distinct: dict[int, float] = {}
            for b in getattr(train_data, "buckets", None) or []:
                res = int(getattr(b, "resolution", 0) or 0)
                if res <= 0:
                    continue
                weight[res] = weight.get(res, 0.0) + len(b)
                d = sum(
                    int(getattr(sub, "_effective_len", 0)) or len(sub)
                    for sub in getattr(b, "datasets", []) or []
                )
                distinct[res] = distinct.get(res, 0.0) + (d or len(b))
            if not weight:
                return
            if schedule_active and getattr(train_data, "_schedule_stages", None):
                stage_res = [s[0] for s in train_data._schedule_stages]
                cum = list(train_data._schedule_cum_frac)
            else:
                stage_res = [frozenset(weight)]
                cum = [1.0]
            stages = schedule_stage_spans(stage_res, cum, total_budget_steps)
            exposure = estimate_image_exposure(stages, weight, distinct, global_batch_size)
            print(format_exposure_report(exposure, target=config.get("min_image_exposure")), flush=True)
            print(
                "rengu_flow: exposure is an estimate (single-resolution batches cycling the "
                "active pool); to even it out, give under-trained resolutions more schedule "
                "stages/fraction or raise epochs.",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - reporting must never break training
            print(f"rengu_flow: could not estimate image exposure: {e}", flush=True)

    _log_resolution_exposure()

    progress_tracker = TrainingProgressTracker(
        max_steps=max_steps,
        total_steps=total_steps,
        loss_window=steps_per_epoch,
    )
    # Throttled stdout progress markers (rank 0). Replaces per-step log spam and
    # per-iteration status.json writes; the web UI parses these for its live bar.
    progress_emitter = ProgressEmitter() if is_main_process() else None
    final_model_name = None
    saved = False
    last_save_step = -1  # last step an inference model was exported (for the final export)
    last_checkpoint_step = -1  # last step a resume checkpoint was written (for the final ckpt)
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
    if isinstance(config.get("micro_batch_size_per_gpu"), dict):
        # Keep bench samples/s honest under per-resolution batches (per-rank share
        # of the real per-step average).
        per_step_batch = max(1, global_batch_size // max(1, world_size_for_opt))

    if resume_from_checkpoint:
        load_lr = "force_constant_lr" not in config and not args.reset_optimizer and not args.reset_optimizer_params
        # WSD rebuilds its schedule for the (possibly extended) horizon and fast-forwards to the
        # resumed step below — loading the saved scheduler state would restore the OLD decay
        # milestone and misplace the tail when extending. Skip the state load for wsd.
        resume_is_wsd = config.get("lr_scheduler", "constant").lower() == "wsd"
        if resume_is_wsd:
            load_lr = False
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
            # Restore the global RNG so post-resume augmentation/dropout/shuffling reproduce the
            # uninterrupted run's stochastic stream (exact for dataloader_num_workers=0).
            from rengu_flow.utils.rng_state import restore_rng_state

            restore_rng_state(client_state.get("rng_state"))
            step = client_state["step"] + 1
            examples = client_state.get("examples", (step - 1) * global_batch_size) + global_batch_size
            epoch = epoch_schedule.current(step)
            if is_main_process():
                print(f"Resuming from checkpoint at epoch {epoch}, step {step}")
            # WSD: position the freshly-built schedule at the resumed step. The stable phase is
            # constant, so this re-anchors the decay tail to the (possibly new) horizon end —
            # resuming the 'predecay' fork with a larger length extends the flat phase instead of
            # re-dropping an already-lowered LR. Cheap arithmetic, even for large step counts.
            if resume_is_wsd and model_engine.lr_scheduler is not None:
                for _ in range(max(0, step - 1)):
                    model_engine.lr_scheduler.step()

    # Lifecycle event (rank 0): one per process start. `resumed` carries the restored step; a
    # restart from step 1 in a folder that already saw a prior run is `restarted_from_scratch`
    # (the timeline itself tells them apart); a genuinely fresh run is `run_started`.
    if is_main_process():
        prior = read_events(run_dir)
        had_prior_run = any(
            e.get("type") in (EVENT_RUN_STARTED, EVENT_RESUMED, EVENT_RESTARTED_FROM_SCRATCH)
            for e in prior
        )
        if resume_from_checkpoint and step > 1:
            sink.event(
                EVENT_RESUMED,
                step=step,
                payload={"checkpoint_tag": resume_tag, "examples": examples},
            )
        elif had_prior_run:
            sink.event(
                EVENT_RESTARTED_FROM_SCRATCH,
                step=step,
                payload={"resume_requested": bool(resume_from_checkpoint)},
            )
        else:
            sink.event(
                EVENT_RUN_STARTED,
                step=step,
                payload={"resume_requested": bool(resume_from_checkpoint)},
            )

    if eval_before_first_step and not resume_from_checkpoint and eval_dataloaders:
        empty_cuda_cache()
        if hasattr(optimizer, "train") and callable(optimizer.train):
            optimizer.train()
        evaluate(
            model,
            model_engine,
            eval_dataloaders,
            sink,
            0,
            eval_gradient_accumulation_steps,
            disable_block_swap_for_eval,
            optimizer=optimizer,
        )
        if val_probe_dataloader is not None:
            last_val_metrics = generalization_probe(
                model,
                model_engine,
                val_probe_dataloader,
                train_probe_dataloader,
                sink,
                0,
                eval_gradient_accumulation_steps,
                disable_block_swap_for_eval,
                probe_batches=val_gap_probe_batches,
                optimizer=optimizer,
            )

    if (
        preview_before_first_step
        and not resume_from_checkpoint
        and previews_configured(config)
    ):
        run_previews(
            model,
            config,
            sink,
            0,
            disable_block_swap=disable_block_swap_for_preview,
            optimizer=optimizer,
        )

    oom_skip_cfg = config.get("train", {}).get("oom_skip", {})
    oom_skip_enabled = bool(oom_skip_cfg.get("enabled", False))
    # On reaching max_in_window OOMs (within the 10-step window), raise blocks_to_swap by this step
    # (freeing more base-model VRAM) and keep going, up to num_blocks, instead of aborting — off by default.
    bump_block_swap = bool(oom_skip_cfg.get("bump_block_swap", False))
    bump_block_swap_step = int(oom_skip_cfg.get("bump_block_swap_step", 2))
    # Emergency checkpoint on ANY otherwise-fatal error (OOM, I/O, etc.) so the run is resumable
    # instead of lost. Best-effort: a failed save never masks the original error. Legacy key
    # ``save_checkpoint_on_oom`` is honoured as the default when the broader key is unset.
    _train_cfg = config.get("train", {})
    save_on_error = bool(
        _train_cfg.get("save_checkpoint_on_error", _train_cfg.get("save_checkpoint_on_oom", True))
    )
    from rengu_flow.utils.oom_skip import is_cuda_oom, reset_engine_timers

    oom_skip_state = None
    if oom_skip_enabled:
        from rengu_flow.utils.oom_skip import OomSkipState, handle_oom_skip

        oom_skip_state = OomSkipState(
            max_in_window=int(
                oom_skip_cfg.get("max_in_window", oom_skip_cfg.get("max_consecutive", 3))
            )
        )

    # Budget backoff: activation_memory_budget is a fraction of the saved set,
    # so its byte translation can overshoot on any new model/resolution/batch
    # combination. Instead of crashing the run, lower the budget and recompile
    # until it fits (the configured value is a desired ceiling, not a promise).
    budget_backoff = None
    if (
        config.get("activation_checkpointing") == "auto"
        and config.get("compile")
        and config.get("activation_budget_backoff", True)
    ):
        from rengu_flow.training.activation_budget import (
            BudgetBackoff,
            apply_activation_memory_budget,
            resolve_auto_ac_budget,
        )

        budget_backoff = BudgetBackoff(resolve_auto_ac_budget(config))
        if config["model"].get("transformer_4bit") and config.get("blocks_to_swap") and is_main_process():
            print(
                "[checkpoint] NOTE: activation_checkpointing='auto' + compile with a "
                "block-swapped 4-bit base tends to overshoot and back off repeatedly — "
                "bitsandbytes keeps every packed weight referenced across fwd->bwd, which "
                "the partitioner's budget cannot see. The measured-lean path for this "
                "combination is activation_checkpointing = true with compile = false "
                "(reentrant recompute defaults on; ~half the peak VRAM).",
                flush=True,
            )

    def _apply_budget_backoff(new_budget: float) -> None:
        """Zero grads, drop every compiled graph, re-arm the (lower) budget."""
        # Must precede any `torch.` use: importing the submodule binds `torch`
        # as a local for the whole function, shadowing the enclosing import.
        import torch._dynamo

        if hasattr(model_engine, "zero_grad"):
            model_engine.zero_grad()
        reset_engine_timers(model_engine)
        empty_cuda_cache()
        torch.cuda.ipc_collect()
        torch._dynamo.reset()
        apply_activation_memory_budget(new_budget)
        if is_main_process():
            print(f"[checkpoint] CUDA OOM -> {budget_backoff.describe()}", flush=True)
        sink.scalar("train/activation_budget", new_budget, step)

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
    epoch = epoch_schedule.current(step)

    # Optional env-gated torch profiler (RENGU_PROF_DIR=/path). Skips compile/warmup steps, captures
    # a steady window, and on completion writes a key_averages table + chrome trace. Zero cost off.
    _prof = _maybe_start_profiler() if is_main_process() else None

    try:
        while True:
            model_engine.reset_activation_shape()
            if train_data is not None and hasattr(train_data, "set_training_context"):
                train_data.set_training_context(train_seed, step)
            # TREAD off-ramp: layers with a progress hook (RouteStartLayer) see the run
            # fraction so tread.disable_after_frac can turn routing off for the final
            # stretch (full-sequence re-calibration recovers most of the routed-training
            # quality gap at a fraction of the speed cost).
            for _rl in route_progress_layers:
                _rl.set_training_progress(step / max(total_budget_steps, 1))
            # Step-accurate resolution schedule: switch the active resolution(s) the
            # moment this step crosses a stage boundary, restarting iteration mid-epoch
            # if needed. Must run before pulling this step's micro-batches.
            if schedule_active:
                train_dataloader.refresh_for_step(step)
            # t0 must wrap the data fetch too: get_data_iterator_for_step calls next()
            # eagerly (pipeline.py), so timing from after it excludes the entire dataloader
            # cost from step_time_sec/steps_per_second/eta — silently hiding a real
            # data-bound bottleneck as if the GPU were the whole step (measured: reported
            # ~2.9 steps/sec vs ~1.2 true wall-clock steps/sec on a data-bound run).
            t0 = time.perf_counter()
            iterator = get_data_iterator_for_step(train_dataloader, model_engine)
            skipped_oom = False
            act_offload_ctx = (
                act_offloader.step() if act_offloader is not None else contextlib.nullcontext()
            )
            if oom_skip_enabled or budget_backoff is not None:
                try:
                    with act_offload_ctx:
                        loss = model_engine.train_batch(iterator).item()
                except Exception as e:
                    if not is_cuda_oom(e):
                        raise
                    # Budget backoff first: with activation_checkpointing="auto" an
                    # OOM usually means the budget's byte translation overshot on
                    # this hardware/config — lowering it and recompiling fixes the
                    # RUN, while oom_skip would just re-OOM every large step.
                    new_budget = budget_backoff.on_oom() if budget_backoff is not None else None
                    if new_budget is not None:
                        _apply_budget_backoff(new_budget)
                        train_dataloader.sync_epoch()
                        skipped_oom = True
                    elif oom_skip_enabled:
                        oom_skip_state.record_skip(step)  # count first so the banner reads N/max
                        handle_oom_skip(
                            oom_skip_state,
                            model_engine,
                            clear_cache=bool(oom_skip_cfg.get("clear_cache_on_skip", True)),
                            step=step,
                            sink=sink,
                        )
                        # Enough OOMs inside the 10-step window? If bump_block_swap is on and the
                        # offloader can still swap more blocks, raise blocks_to_swap and start a
                        # fresh window (retry at the higher swap) instead of aborting; only abort
                        # once every block is already swapped (or bump is off).
                        if oom_skip_state.at_limit(step):
                            _off = getattr(model, "_block_swap_offloader", None)
                            _bumped = False
                            if (
                                bump_block_swap
                                and _off is not None and getattr(_off, "enabled", False)
                            ):
                                _before = _off.blocks_to_swap
                                _after = _off.increase_swap(bump_block_swap_step)
                                if _after > _before:
                                    oom_skip_state.reset_window()  # fresh window at higher swap
                                    _bumped = True
                                    if is_main_process():
                                        print(
                                            f"rengu_flow: {oom_skip_state.max_in_window} OOMs within "
                                            f"{oom_skip_state.window} steps — raised blocks_to_swap "
                                            f"{_before} -> {_after}, retrying (oom_skip.bump_block_swap)",
                                            flush=True,
                                        )
                            if not _bumped:
                                raise RuntimeError(
                                    f"OOM {oom_skip_state.max_in_window} times within "
                                    f"{oom_skip_state.window} training steps, aborting training"
                                )
                        train_dataloader.sync_epoch()
                        skipped_oom = True
                    else:
                        raise
            else:
                with act_offload_ctx:
                    loss = model_engine.train_batch(iterator).item()
            iter_sec = time.perf_counter() - t0
            if _prof is not None:
                _prof.step()
            if skipped_oom:
                # An OOM-skipped step still consumes one step of the budget.
                if step >= total_budget_steps:
                    final_model_name, _reason = budget_reached_target(max_steps, epochs, step)
                    if is_main_process():
                        print(_reason)
                    break
                step += 1
                examples += global_batch_size
                continue
            if training_ema is not None and step % training_ema.update_interval == 0:
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
            # Epoch boundary, save naming and save/eval cadence all come from the single
            # EpochSchedule authority: `completed_at` is the epoch that finished exactly at this
            # step (so the export is named after the COMPLETED epoch — first is epoch1, not epoch2).
            completed_epoch = epoch_schedule.completed_at(step)
            finished_epoch = completed_epoch is not None
            if finished_epoch:
                checkpointed_ep, saved_ep = saver.process_epoch_boundary(
                    completed_epoch, step, examples
                )
                if checkpointed_ep:
                    last_checkpoint_step = step
                if saved_ep:
                    saved = True
                    last_save_step = step
            epoch = epoch_schedule.current(step)

            x_axis = examples if x_axis_examples else step
            progress_tracker.record_step_duration(iter_sec)
            progress_tracker.record_loss(loss)
            step_progress = progress_tracker.metrics(step=step)
            # Throttled progress marker to stdout (rank 0). Always emit on the final step and
            # on save/epoch boundaries so the UI never misses a transition; otherwise the
            # emitter caps the rate (~1/sec). `epoch` is the EpochSchedule's budget epoch, so
            # it stays meaningful (1..epochs) even with short resolution-staged dataloader epochs.
            if progress_emitter is not None:
                is_final = step >= total_budget_steps
                progress_emitter.emit(
                    build_progress_payload(
                        step=step,
                        loss=loss,
                        epoch=epoch,
                        metrics=step_progress,
                        val_metrics=last_val_metrics,
                    ),
                    force=is_final or finished_epoch or saved,
                )
            log_training_step(
                sink=sink,
                optimizer=optimizer,
                loss=loss,
                x_axis=x_axis,
                step=step,
                logging_steps=logging_steps,
                is_main=is_main_process(),
                model_engine=model_engine,
            )

            if (eval_every_n_steps and step % eval_every_n_steps == 0) or (
                finished_epoch and eval_every_n_epochs and completed_epoch % eval_every_n_epochs == 0
            ):
                evaluate(
                    model,
                    model_engine,
                    eval_dataloaders,
                    sink,
                    x_axis,
                    eval_gradient_accumulation_steps,
                    disable_block_swap_for_eval,
                    optimizer=optimizer,
                )
                if val_probe_dataloader is not None:
                    last_val_metrics = generalization_probe(
                        model,
                        model_engine,
                        val_probe_dataloader,
                        train_probe_dataloader,
                        sink,
                        x_axis,
                        eval_gradient_accumulation_steps,
                        disable_block_swap_for_eval,
                        probe_batches=val_gap_probe_batches,
                        optimizer=optimizer,
                    )
                    # Re-emit a fresh marker so the UI surfaces the new val/gap promptly
                    # (the per-step emit above ran before this probe).
                    if progress_emitter is not None and last_val_metrics is not None:
                        progress_emitter.emit(
                            build_progress_payload(
                                step=step,
                                loss=loss,
                                epoch=epoch,
                                metrics=step_progress,
                                val_metrics=last_val_metrics,
                            ),
                            force=True,
                        )

            if finished_epoch:
                if is_main_process() and num_steps > 0:
                    avg_epoch_loss = epoch_loss / num_steps
                    sink.scalar("train/epoch_loss", avg_epoch_loss, epoch)
                epoch_loss = 0.0
                num_steps = 0

            step_checkpointed, step_saved, step_signals = saver.process_step(step, examples)
            if step_checkpointed:
                last_checkpoint_step = step
            if step_saved:
                saved = True
                last_save_step = step

            # WSD: at the decay onset, save the protected pre-decay fork once (LR still at base).
            if wsd_fork_step is not None and step >= wsd_fork_step:
                if saver.save_fork_checkpoint(step, examples) and is_main_process():
                    print(
                        f"rengu_flow: WSD pre-decay fork saved at step {step} (tag 'predecay'); "
                        "extend later by resuming from it with a larger epochs/max_steps."
                    )
                wsd_fork_step = None  # fire once

            # Hot-reload the [preview] section from the config file on a `reload_config`
            # signal (edit the TOML, then signal). Applies live to the checks below; only
            # [preview] is reloaded — model/optimizer/dataset can't change mid-run.
            if step_signals.should_reload_config:
                reload_preview_config(config, args.config, sink=sink, step=step)

            preview_x_axis = examples if x_axis_examples else step
            forced_preview = step_signals.should_preview
            if should_run_previews(
                config,
                step,
                epoch,
                finished_epoch=finished_epoch,
                forced=forced_preview,
            ):
                run_previews(
                    model,
                    config,
                    sink,
                    preview_x_axis,
                    disable_block_swap=disable_block_swap_for_preview,
                    optimizer=optimizer,
                )
            elif forced_preview and is_main_process():
                # A preview was explicitly requested but there are no prompts to render. Don't
                # silently swallow it — say so, so the log explains why no image appeared.
                print(
                    "rengu_flow: preview signal received but no prompts are configured "
                    "([preview].prompts is empty) — nothing to render",
                    flush=True,
                )

            # Bound the block-swap allocator creep (see block_swap_active above): reclaim the
            # reserved pool periodically when neither eval nor preview is doing it for us. The
            # single-buffered swap path (and eager fp8/AC transients under an adapter) allocate
            # many shape-varying tensors per step, fragmenting the *reserved* pool by hundreds of
            # MB between reclaims — enough to OOM a tight blocks_to_swap. A 50-step cadence let it
            # grow ~0.3-0.8 GB; reclaim more often (default every 10) so the pool stays bounded and
            # tight swap settings don't OOM. Bare empty_cache() is cheap; keep the full
            # gc.collect()+empty_cache() on the sparse 50-step cadence.
            reclaim_every = int(config.get("block_swap_reclaim_every", 10))
            if block_swap_active and step > 0 and reclaim_every > 0 and step % reclaim_every == 0:
                if step % 50 == 0:
                    empty_cuda_cache()
                else:
                    torch.cuda.empty_cache()

            if step >= total_budget_steps:
                final_model_name, _reason = budget_reached_target(max_steps, epochs, step)
                if is_main_process():
                    print(_reason)
                break
            step += 1
            examples += global_batch_size
    except SystemExit:
        # Graceful stop: the saver received save_quit/export_quit, already checkpointed and
        # printed the reason, then sys.exit(0). Record it on the timeline and flush tracking
        # before the process exits (SystemExit is not caught by the `except Exception` below).
        if sampler is not None:
            sampler.stop()
        sink.event(EVENT_STOP_REQUESTED, step=step)
        sink.close(status="stopped")
        raise
    except Exception as exc:
        # Last-ditch checkpoint on ANY otherwise-fatal error (OOM during a preview/eval, an I/O
        # error, etc.) so the run can resume instead of being lost. Best-effort and rank-0 gated
        # logging; the original error is always re-raised. Note: in multi-GPU runs the save's
        # collective may not complete if only some ranks failed — the re-raise still tears down.
        if save_on_error and last_checkpoint_step != step:
            reason = "CUDA OOM" if is_cuda_oom(exc) else f"error ({type(exc).__name__})"
            if is_main_process():
                print(
                    f"rengu_flow: fatal {reason} at step {step} — attempting emergency checkpoint "
                    "before exit...",
                    flush=True,
                )
            try:
                empty_cuda_cache()
                if saver.save_checkpoint(step, examples):
                    last_checkpoint_step = step
                    if is_main_process():
                        print(
                            f"rengu_flow: emergency checkpoint saved at step {step}; resume with "
                            "--resume_from_checkpoint.",
                            flush=True,
                        )
            except Exception as save_exc:  # noqa: BLE001 - never mask the original error
                if is_main_process():
                    print(
                        f"rengu_flow: emergency checkpoint FAILED after {reason}: {save_exc}",
                        flush=True,
                    )
        if sampler is not None:
            sampler.stop()
        sink.event(EVENT_FAILED, step=step, payload={"error": str(exc)[:500]})
        sink.close(status="failed")
        raise

    # End-of-run saves (decision is the unit-tested plan_final_saves): always write a final
    # resume checkpoint so the run can continue from the exact last step, and the final model,
    # each unless one was already written at this very step.
    write_checkpoint, export_name = plan_final_saves(
        step=step,
        last_checkpoint_step=last_checkpoint_step,
        last_save_step=last_save_step,
        final_model_name=final_model_name,
    )
    if write_checkpoint and saver.save_checkpoint(step, examples):
        last_checkpoint_step = step
    if export_name:
        saver.save_model(export_name)
    saver.shutdown_async_exports()

    if is_main_process():
        if bench_enabled(config):
            bench_summarize(
                bench_csv,
                config.get("run_name", "bench"),
                run_dir,
            )
        if config.get("compile"):
            _print_compile_cache_stats()
        print("Training complete.")

    if sampler is not None:
        sampler.stop()
    sink.event(EVENT_FINISHED, step=step)
    sink.close(status="finished")


def _print_compile_cache_stats() -> None:
    """Summarize Inductor disk-cache effectiveness for this run.

    Misses on a config that ran before mean the cache got re-keyed: the key
    hashes the compile-relevant config (activation_checkpointing mode,
    activation_memory_budget, compile_mode, torch version) plus every tensor
    shape in the graph (resolution buckets, micro-batch, adapter rank/factor).
    Optimizer settings are NOT part of the key (the optimizer step is not
    compiled)."""
    try:
        from torch._dynamo.utils import counters

        hits = counters["inductor"].get("fxgraph_cache_hit", 0)
        misses = counters["inductor"].get("fxgraph_cache_miss", 0)
    except Exception:
        return
    if hits or misses:
        msg = f"[compile-cache] fxgraph disk cache: {hits} hits / {misses} misses this run."
        if misses and not hits:
            msg += (
                " All misses on a config you ran before means the cache was re-keyed —"
                " caused by changing activation_checkpointing/activation_memory_budget/"
                "compile_mode, adapter rank/factor, micro_batch size, resolutions, or a"
                " torch upgrade. Optimizer changes do NOT re-key it."
            )
        print(msg, flush=True)


def run_prepared(args) -> None:
    if args.dump_dataset is not None:
        from rengu_flow.data.dump_dataset import dump_dataset

        dump_dataset(args.dump_dataset)
        return
    if args.config is None:
        raise SystemExit("rengu_flow: --config is required (unless using --dump_dataset).")

    config = load_config(args.config)
    # Smoke/CI fixtures omit [model] paths; the smoke scripts export RENGU_*_PATH
    # from the repo-root .env before launching. Only an externally-set env var
    # overrides — the trainer itself never reads .env, so normal runs are
    # unaffected.
    apply_model_paths_from_env(config)
    set_config_defaults(config)

    if not args.validate_only:
        dataset_config = load_dataset_config(config)
        if dataset_config is not None:
            config["_dataset_config_loaded"] = dataset_config

    try:
        validate_config(config, for_script=True)
        from rengu_flow.config.preflight import collect_preflight_issues
        from rengu_flow.config.validation import format_validation_issues

        preflight = collect_preflight_issues(config)
        if preflight:
            raise ConfigValidationError(format_validation_issues(preflight))
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

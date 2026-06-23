"""Multi-GPU DeepSpeed pipeline engine ("deepspeed")."""
from __future__ import annotations

import sys
from shutil import which

from rengu_flow.engine.base import TrainingBackend


class DeepSpeedPipeBackend(TrainingBackend):
    name = "deepspeed"

    @classmethod
    def launch_argv(cls, config, *, config_path, num_gpus, master_port):
        deepspeed = which("deepspeed")
        if not deepspeed:  # fall back like today when the launcher is absent
            return [sys.executable, "-m", "rengu_flow.main", "--config", str(config_path)]
        cmd = [deepspeed, f"--num_gpus={num_gpus}"]
        if master_port is not None:
            cmd.append(f"--master_port={master_port}")
        cmd += ["--module", "rengu_flow.main", "--config", str(config_path)]
        return cmd

    def validate(self, config):
        # Block-swap guards that apply to the DeepSpeed engine.
        if config.get("blocks_to_swap", 0):
            if config.get("pipeline_stages", 1) != 1:
                raise ValueError("Block swapping requires pipeline_stages = 1.")
            if not bool(config.get("adapter")) and not config.get("optimizer", {}).get("gradient_release"):
                raise ValueError(
                    "Block swapping for full-model training requires optimizer.gradient_release = true "
                    "(the per-parameter optimizer step must run during the backward pass while each "
                    "block is on the GPU)."
                )

    @property
    def is_distributed(self): return True

    @property
    def supports_block_swap(self): return True

    @property
    def supports_gradient_release(self): return True

    def build_pipe(self, *, layers, num_stages, partition_method, manual_partition_split, loss_fn, extra_kw):
        from rengu_flow.utils.pipeline import ManualPipelineModule
        return ManualPipelineModule(
            layers=layers, num_stages=num_stages, partition_method=partition_method,
            manual_partition_split=manual_partition_split, loss_fn=loss_fn, **extra_kw,
        )

    def build_engine(self, *, pipeline_model, ds_config, args, get_optimizer, parameters_to_train):
        import deepspeed
        engine, _, _, _ = deepspeed.initialize(args=args, model=pipeline_model, config=ds_config)
        engine._support_torch_style_backward = True
        engine._configure_optimizer(get_optimizer, parameters_to_train)
        return engine

    def make_cache_worker(self, cache_fn, args):
        # Multi-GPU: rank-0 process worker feeds a Manager queue broadcast to all ranks.
        try:
            import multiprocess as mp
        except ImportError:
            import multiprocessing as mp
        manager = mp.Manager()
        q = manager.Queue()
        worker = mp.Process(target=cache_fn, args=(args, q))
        # Pin the Manager to the worker so its process stays alive as long as the queue is used:
        # DatasetManager.cache() holds the worker until worker.join(). Without this, the local
        # `manager` would be dropped on return and its process could be GC-shut-down, invalidating
        # the broadcast queue proxy on the other ranks.
        worker._cache_mp_manager = manager
        return worker, q

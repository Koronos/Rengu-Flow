"""DeepSpeed pipeline module and data iterator helper.

DeepSpeed is imported lazily: ``ManualPipelineModule`` is built on first attribute access
(module-level ``__getattr__``, PEP 562) so the data-iterator helper ``get_data_iterator_for_step``
— used by BOTH the DeepSpeed engine and the single-GPU torch engine — can be imported on a host
without DeepSpeed installed (native Windows, engine='accelerate')."""

from itertools import islice

from torch import nn

_MANUAL_PIPELINE_MODULE_CLS = None


def _build_manual_pipeline_module_cls():
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.pipe import LayerSpec

    class ManualPipelineModule(PipelineModule):
        """PipelineModule with manual partition_split for uneven layer distribution across GPUs."""

        def __init__(self, *args, manual_partition_split=None, **kwargs):
            self.manual_partition_split = manual_partition_split
            super().__init__(*args, **kwargs)

        def _partition_layers(self, method="uniform"):
            if method.lower() == "manual" and self.manual_partition_split is not None:
                num_stages = self._topo.get_dim("pipe")
                stage_id = self._topo.get_coord(self.global_rank).pipe
                num_partitions = len(self.manual_partition_split)
                assert num_partitions == num_stages - 1, (
                    f"partition_split must have length {num_stages - 1} (pipeline_stages - 1), got {num_partitions}"
                )
                total_layers = len(self._layer_specs)
                boundaries = [0] + list(self.manual_partition_split) + [total_layers]
                self.parts = boundaries
                if self.global_rank == 0:
                    for stage in range(num_stages):
                        start = self.parts[stage]
                        stop = self.parts[stage + 1]
                        print(f"stage={stage} layers={stop - start}")
                        for idx, layer in enumerate(self._layer_specs[start:stop]):
                            name = str(layer)
                            if isinstance(layer, LayerSpec):
                                name = layer.typename.__name__
                            elif isinstance(layer, nn.Module):
                                name = layer.__class__.__name__
                            else:
                                try:
                                    name = layer.__name__
                                except AttributeError:
                                    pass
                            print(f"    {idx + start:2d}: {name}")
                    if self.loss_fn:
                        try:
                            print(f"  loss: {self.loss_fn.__name__}")
                        except AttributeError:
                            print(f"  loss: {self.loss_fn.__class__.__name__}")
                self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
            else:
                super()._partition_layers(method)

    return ManualPipelineModule


def __getattr__(name):  # PEP 562: lazy DeepSpeed import only when the class is actually used.
    if name == "ManualPipelineModule":
        global _MANUAL_PIPELINE_MODULE_CLS
        if _MANUAL_PIPELINE_MODULE_CLS is None:
            _MANUAL_PIPELINE_MODULE_CLS = _build_manual_pipeline_module_cls()
        return _MANUAL_PIPELINE_MODULE_CLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    """Return one step of micro-batches, preloading only when the engine requires pipeline IPC."""
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    if not getattr(engine, "preload_micro_batches", True):
        return islice(dataloader_iter, num_micro_batches)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)

"""Utilities for Rengu Flow (common, pipeline, etc.).

Submodules are attached lazily (Scientific-Python SPEC 1 / ``lazy_loader``): ``pipeline`` eager-imports
DeepSpeed's ``PipelineModule`` (a ~17s import), so importing other utils — which nearly the whole
codebase does transitively — must not pull it in. Accessing a pipeline symbol triggers its import
exactly when a caller needs it.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "common": ["get_rank", "is_main_process"],
        "pipeline": ["ManualPipelineModule", "get_data_iterator_for_step"],
    },
)

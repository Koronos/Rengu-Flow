"""Data loading for training: Dataset, DatasetManager, PipelineDataLoader, synthetic dataset.

Submodules are attached lazily (Scientific-Python SPEC 1 / ``lazy_loader``) so importing one data
helper (e.g. ``Dataset`` for config/validation work) does not eagerly pull every sibling module.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "dataset": ["Dataset"],
        "loader": ["PipelineDataLoader", "split_batch"],
        "manager": ["DatasetManager"],
        "synthetic": ["SyntheticSDXLDataset"],
        "augmentation": [
            "AugmentationConfigError",
            "AugmentationStrategyNotImplementedError",
        ],
        "dataset_config": [
            "DatasetConfigError",
            "validate_dataset_config_for_real_data",
        ],
    },
)

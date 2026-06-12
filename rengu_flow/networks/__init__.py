"""Network adapters for training (LoRA, LoKr, etc.) per model. One module per network+model.

Adapters are attached lazily (Scientific-Python SPEC 1 / ``lazy_loader``): each pulls heavy deps
(``peft`` / ``diffusers``), and a given model only uses its own adapters. Importing the package — or
a model that references ``networks.<adapter>`` at runtime — therefore must not eager-import every
adapter (e.g. importing SDXL would otherwise pull ``adapter_dit`` -> ``peft``, which SDXL never uses).
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["adapter_dit", "lokr_sdxl", "lora_sdxl", "lycoris_attach", "lycoris_meta", "lycoris_sdxl"],
    submod_attrs={"factorization": ["factorization"]},
)

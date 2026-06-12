"""Dataset preparation: tagging, captioning, watermark cleanup, and tag editing.

Runs outside the training path (own CLI stage processes / UI section). Submodules are
attached lazily (SPEC 1) so importing one helper never pulls heavy inference deps.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "caption_store": ["CaptionSet", "CaptionStore"],
        "tag_ops": ["TagEditOp", "TagFilter", "apply_ops", "tag_frequencies"],
    },
)

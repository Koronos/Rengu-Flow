"""Optimizers vendored from diffusion-pipe. See README.md and NOTICE.md.

Submodules are imported on demand (registry lazy-load) to avoid optional deps at import time.
"""

from .gradient_release import GradientReleaseOptimizerWrapper

__all__ = ["GradientReleaseOptimizerWrapper"]

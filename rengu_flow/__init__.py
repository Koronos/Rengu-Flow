"""Rengu Flow: modular training framework with TOML config and registry-based components."""

# Neutralize WSL-toxic CUDA allocator settings at the earliest possible point — before any
# submodule (and therefore torch) is imported, since the caching allocator parses
# PYTORCH_CUDA_ALLOC_CONF when torch is first imported. platform_compat is stdlib-only at import
# time, so this stays light and has no effect off WSL.
from rengu_flow.platform_compat import configure_cuda_allocator as _configure_cuda_allocator

_configure_cuda_allocator()
del _configure_cuda_allocator

# Single source of truth: read the version back from package metadata (pyproject [project].version)
# instead of hardcoding it here. See rengu_flow/version.py.
from rengu_flow.version import package_version as _package_version

__version__ = _package_version()

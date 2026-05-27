"""Augmentation configuration and runtime errors."""

from __future__ import annotations


class AugmentationError(ValueError):
    """Base class for augmentation errors."""


class AugmentationConfigError(AugmentationError):
    """Invalid augmentation TOML or merge result."""


class AugmentationStrategyNotImplementedError(AugmentationError):
    """Known strategy name that is not implemented in this build."""

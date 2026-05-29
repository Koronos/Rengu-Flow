"""Dataset image augmentation (MVP Tier A–B)."""

from rengu_flow.data.augmentation.apply import (
    apply_augmentation,
    augmentation_seed_for_image,
)
from rengu_flow.data.augmentation.branches import expand_variant_keys
from rengu_flow.data.augmentation.config import (
    augmentation_fingerprint,
    is_augmentation_enabled,
    resolve_augmentation_config,
    validate_augmentation_for_directory,
)
from rengu_flow.data.augmentation.errors import (
    AugmentationConfigError,
    AugmentationStrategyNotImplementedError,
)
from rengu_flow.data.augmentation.names import (
    AUG_MVP_VERSION,
    IMPLEMENTED_STRATEGIES,
    MVP_PRESET_NAMES,
)
from rengu_flow.data.augmentation.spec_utils import (
    image_spec_base,
    image_spec_variant_key,
    with_variant_key,
)

__all__ = [
    "AUG_MVP_VERSION",
    "AugmentationConfigError",
    "AugmentationStrategyNotImplementedError",
    "IMPLEMENTED_STRATEGIES",
    "MVP_PRESET_NAMES",
    "apply_augmentation",
    "augmentation_fingerprint",
    "augmentation_seed_for_image",
    "expand_variant_keys",
    "image_spec_base",
    "image_spec_variant_key",
    "is_augmentation_enabled",
    "resolve_augmentation_config",
    "validate_augmentation_for_directory",
    "with_variant_key",
]

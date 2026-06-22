"""Tests for training-config dataset library ref strings."""

import pytest

from rengu_flow.config.dataset_library_ref import (
    canonical_dataset_ref,
    dataset_library_ref,
    is_library_dataset_ref,
    library_dataset_id_from_ref,
    library_dataset_label_from_ref,
)


def test_ref_with_display_suffix() -> None:
    ref = "rengu-flow-dataset:3:artista 1"
    assert is_library_dataset_ref(ref)
    assert library_dataset_id_from_ref(ref) == 3
    assert library_dataset_label_from_ref(ref) == "artista 1"
    assert canonical_dataset_ref(ref) == "rengu-flow-dataset:3"


def test_ref_without_suffix() -> None:
    ref = "rengu-flow-dataset:12"
    assert library_dataset_id_from_ref(ref) == 12
    assert library_dataset_label_from_ref(ref) is None


def test_format_ref() -> None:
    assert dataset_library_ref(5, "My set") == "rengu-flow-dataset:5:My set"
    assert dataset_library_ref(5) == "rengu-flow-dataset:5"


def test_invalid_ref() -> None:
    with pytest.raises(ValueError):
        library_dataset_id_from_ref("rengu-flow-dataset:abc")

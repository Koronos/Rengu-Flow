"""Dataset form schema help coverage."""

from renga_flow_ui.dataset_schema import get_dataset_schema


def test_all_dataset_fields_have_help() -> None:
    schema = get_dataset_schema()
    missing = []
    for sec in schema["sections"]:
        for field in sec.get("fields", []):
            path = field.get("path")
            if not path:
                continue
            if not field.get("help"):
                missing.append(path)
    assert missing == [], f"Fields without help: {missing}"


def test_augmentation_directory_fields_have_help() -> None:
    schema = get_dataset_schema()
    missing = []
    for field in schema.get("augmentation_directory_fields", []):
        path = field.get("path")
        if not path:
            continue
        if not field.get("help"):
            missing.append(path)
    assert missing == [], f"Augmentation fields without help: {missing}"

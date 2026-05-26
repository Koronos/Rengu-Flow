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

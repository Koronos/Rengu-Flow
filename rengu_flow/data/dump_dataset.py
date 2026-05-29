"""Dump dataset metadata (paths, captions) without loading a model or GPU."""

from __future__ import annotations

import json
from pathlib import Path

import toml

from rengu_flow.data.dataset import CAPTIONS_JSON_FILE, _read_captions_from_txt_per_line
from rengu_flow.data.dataset_config import validate_dataset_config_for_real_data


def _resolve_path(dataset_file: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd = Path.cwd()
    if (cwd / p).exists():
        return cwd / p
    return dataset_file.parent / p


def _caption_for_image(image_path: Path, directory_config: dict, captions_json: dict | None) -> list[str]:
    key = image_path.name
    if captions_json is not None:
        caps = captions_json.get(key)
        if caps is None:
            return [""]
        return caps if isinstance(caps, list) else [str(caps)]
    txt = image_path.with_suffix(".txt")
    if txt.is_file():
        return _read_captions_from_txt_per_line(str(txt))
    directory_caption = directory_config.get("directory_caption")
    if directory_caption is not None:
        return [directory_caption]
    return [""]


def dump_dataset(dataset_path: Path) -> list[dict]:
    """Print JSON lines per image and return the same records."""
    dataset_path = Path(dataset_path)
    with open(dataset_path) as f:
        dataset_config = toml.load(f)
    validate_dataset_config_for_real_data(dataset_config)

    records: list[dict] = []
    for directory_config in dataset_config["directory"]:
        dir_path = _resolve_path(dataset_path, directory_config["path"])
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

        captions_json_path = dir_path / CAPTIONS_JSON_FILE
        captions_json = None
        if captions_json_path.is_file():
            with open(captions_json_path) as f:
                captions_json = json.load(f)

        skip_suffixes = {".txt", ".npz", ".json", ".parquet", ".bak"}
        for file in sorted(dir_path.iterdir()):
            if not file.is_file() or file.suffix.lower() in skip_suffixes:
                continue
            captions = _caption_for_image(file, directory_config, captions_json)
            record = {
                "directory": str(dir_path),
                "image": str(file),
                "captions": captions,
                "num_repeats": directory_config.get("num_repeats", 1),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))

    return records

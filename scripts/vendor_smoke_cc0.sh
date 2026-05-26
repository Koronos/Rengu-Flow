#!/usr/bin/env bash
# Vendor 12 CC0 GB82 images into tests/fixtures/smoke_cc0/images/ (JPEG 512px + captions).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="${REPO_ROOT}/tests/fixtures/smoke_cc0"
IMAGES_DIR="${FIXTURE_DIR}/images"
MANIFEST="${FIXTURE_DIR}/manifest.json"
# Pinned upstream commit (gb82-image-set); do not use floating main.
export GB82_COMMIT="502f9f94cb73d1ad5c89ce06fe6b100d0a27df8f"
GB82_REPO="https://github.com/gianni-rosato/gb82-image-set.git"
WORK_DIR="${TMPDIR:-/tmp}/renga_flow_gb82_vendor_$$"

cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${IMAGES_DIR}"
mkdir -p "${WORK_DIR}"

if [[ ! -d "${WORK_DIR}/gb82/.git" ]]; then
  git clone --depth 1 "${GB82_REPO}" "${WORK_DIR}/gb82"
  git -C "${WORK_DIR}/gb82" fetch --depth 1 origin "${GB82_COMMIT}"
  git -C "${WORK_DIR}/gb82" checkout "${GB82_COMMIT}"
fi

PNG_DIR="${WORK_DIR}/gb82/png"
if [[ ! -d "${PNG_DIR}" ]]; then
  echo "Expected png/ in gb82 checkout at ${PNG_DIR}" >&2
  exit 1
fi

cp "${WORK_DIR}/gb82/LICENSE" "${FIXTURE_DIR}/LICENSE"

export MANIFEST PNG_DIR IMAGES_DIR FIXTURE_DIR
python3 <<'PY'
import json
import os
from datetime import date
from pathlib import Path

from PIL import Image

manifest_path = Path(os.environ["MANIFEST"])
png_dir = Path(os.environ["PNG_DIR"])
images_dir = Path(os.environ["IMAGES_DIR"])
fixture_dir = Path(os.environ["FIXTURE_DIR"])
commit = os.environ["GB82_COMMIT"]

entries = json.loads(manifest_path.read_text())
images_dir.mkdir(parents=True, exist_ok=True)

stems = []
for entry in entries:
    src = png_dir / entry["upstream_png"]
    if not src.is_file():
        raise FileNotFoundError(src)
    stem = entry["output_stem"]
    stems.append(stem)
    img = Image.open(src).convert("RGB")
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    img.save(images_dir / f"{stem}.jpg", format="JPEG", quality=90, optimize=True)
    (images_dir / f"{stem}.txt").write_text(entry["caption"].strip() + "\n", encoding="utf-8")

lines = [
    "# smoke_cc0 — GB82 subset",
    "",
    f"Generated: {date.today().isoformat()}",
    "",
    f"Source: [gianni-rosato/gb82-image-set](https://github.com/gianni-rosato/gb82-image-set) "
    f"(commit `{commit}`).",
    "",
    "Upstream license: CC0 1.0 Universal (see LICENSE in this directory).",
    "",
    "Twelve JPEG images (max edge 512px, quality 90) with project-written one-line `.txt` "
    "captions from `manifest.json` (not copied from third-party metadata).",
    "",
    "## Files",
    "",
]
lines.extend(f"- `{stem}.jpg` / `{stem}.txt`" for stem in stems)
(fixture_dir / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Vendored {len(entries)} image pairs into {images_dir}")
PY

echo "Done: ${IMAGES_DIR}"

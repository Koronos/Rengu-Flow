"""ToriiGate captioning via llama.cpp (GGUF) — the fast path for ToriiGate.

ToriiGate-0.5 is a hybrid linear-attention VLM (Qwen3.5). transformers runs its linear
layers in a slow Python fallback (the model card itself warns transformers is "extremely
slow"); llama.cpp has the optimized kernels. This backend downloads, on demand:

  - a pinned llama.cpp **Vulkan** release binary (GPU on any vendor, no CUDA toolchain), and
  - the community GGUF + vision projector (mmproj),

then runs ``llama-server`` (model loaded once, continuous batching across slots) and
captions the whole folder over its OpenAI-compatible endpoint. Validated on an RTX 4080:
~58 img/min at Q8_0 (~lossless), ~8 GB VRAM — vs the transformers path's "extremely slow".

Both downloads are on first use only; nothing here touches rengu's venv.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# Pinned llama.cpp release that supports ToriiGate's qwen3_5 vision arch. Bump deliberately.
LLAMACPP_RELEASE = "b9837"
_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{rel}/{asset}"

GGUF_REPO = "DraconicDragon/ToriiGate-0.5-GGUF"
MMPROJ_FILE = "ToriiGate-0.5-fp16.mmproj.gguf"  # fp16 vision projector (small, keep quality)
# Selectable weight quantizations, smallest/fastest -> largest/best. Q8_0 ≈ lossless.
GGUF_QUANTS: dict[str, str] = {
    "Q4_K_M": "ToriiGate-0.5-Q4_K_M.gguf",
    "Q5_K_M": "ToriiGate-0.5-Q5_K_M.gguf",
    "Q6_K": "ToriiGate-0.5-Q6_K.gguf",
    "Q8_0": "ToriiGate-0.5-Q8_0.gguf",
}
DEFAULT_QUANT = "Q8_0"

# Validated server tuning (see the optimization notes in docs/user/dataset-prep.md):
# 16 slots saturates the GPU; defaults beat -fa/-ub overrides; images are capped to ~1 Mpx.
N_PARALLEL = 16
CTX_SIZE = 32768
MAX_PIXELS = 1_000_000


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "rengu-flow" / "llamacpp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _release_asset() -> str:
    """Pick the Vulkan release asset for this platform (GPU without a CUDA match)."""
    from rengu_flow.platform_compat import PLATFORM

    if PLATFORM.is_windows:
        return f"llama-{LLAMACPP_RELEASE}-bin-win-vulkan-x64.zip"
    if PLATFORM.is_macos:
        raise RuntimeError(
            "The GGUF caption engine ships a Vulkan build for Linux/Windows; macOS isn't "
            "supported here. Use engine='hf' for ToriiGate on macOS."
        )
    return f"llama-{LLAMACPP_RELEASE}-bin-ubuntu-vulkan-x64.tar.gz"


def ensure_binary() -> Path:
    """Download + extract the pinned llama.cpp release once; return the dir holding the
    executables and their shared libs (used as both PATH and LD_LIBRARY_PATH)."""
    asset = _release_asset()
    dest = _cache_root() / LLAMACPP_RELEASE
    server = dest / ("llama-server.exe" if asset.endswith(".zip") else "llama-server")
    if server.is_file():
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    url = _RELEASE_URL.format(rel=LLAMACPP_RELEASE, asset=asset)
    archive = dest / asset
    logger.info("Downloading llama.cpp %s (%s) ...", LLAMACPP_RELEASE, asset)
    urllib.request.urlretrieve(url, archive)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    archive.unlink(missing_ok=True)

    # Releases may nest the binaries in a subdir; locate the real llama-server.
    if not server.is_file():
        found = next((p for p in dest.rglob(server.name)), None)
        if found is None:
            raise RuntimeError(f"llama-server not found after extracting {asset}")
        return found.parent
    return dest


def ensure_gguf(quant: str) -> tuple[Path, Path]:
    """Download the chosen weight GGUF + the mmproj (via HF cache); return both paths."""
    from huggingface_hub import hf_hub_download

    fname = GGUF_QUANTS.get(quant) or GGUF_QUANTS[DEFAULT_QUANT]
    gguf = Path(hf_hub_download(repo_id=GGUF_REPO, filename=fname))
    mmproj = Path(hf_hub_download(repo_id=GGUF_REPO, filename=MMPROJ_FILE))
    return gguf, mmproj


def _server_env(bin_dir: Path) -> dict:
    env = dict(os.environ)
    # The release ships its shared libs (libggml*, vulkan backend) next to the binary.
    env["LD_LIBRARY_PATH"] = f"{bin_dir}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _start_server(bin_dir: Path, gguf: Path, mmproj: Path, port: int):
    exe = bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
    cmd = [
        str(exe), "-m", str(gguf), "--mmproj", str(mmproj),
        "-ngl", "99", "-c", str(CTX_SIZE), "--parallel", str(N_PARALLEL),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    logger.info("Starting llama-server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, env=_server_env(bin_dir),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc


def _wait_health(port: int, proc, timeout: float = 180.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become healthy in time")


def _encode_image(path: Path) -> str:
    """Resize to <= MAX_PIXELS (ToriiGate's training res; bounds vision tokens so a request
    can't overflow a server slot) and return base64 JPEG."""
    import base64
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.width * im.height > MAX_PIXELS:
        s = (MAX_PIXELS / (im.width * im.height)) ** 0.5
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _request_caption(port: int, b64: str, prompt: str, config) -> str:
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": config.temperature if config.temperature is not None else 0.5,
        "top_p": config.top_p if config.top_p is not None else 1.0,
        "max_tokens": config.max_new_tokens,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

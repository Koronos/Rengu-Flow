"""VLM captioning via llama.cpp (GGUF): ToriiGate's fast path and the edit-instruction models.

ToriiGate-0.5 is a hybrid linear-attention VLM (Qwen3.5). transformers runs its linear
layers in a slow Python fallback (the model card itself warns transformers is "extremely
slow"); llama.cpp has the optimized kernels. This backend downloads, on demand:

  - a pinned llama.cpp **Vulkan** release binary (GPU on any vendor, no CUDA toolchain), and
  - the community GGUF + vision projector (mmproj),

then runs ``llama-server`` (model loaded once, continuous batching across slots) and
captions the whole folder over its OpenAI-compatible endpoint. Validated on an RTX 4080:
~58 img/min at Q8_0 (~lossless), ~8 GB VRAM — vs the transformers path's "extremely slow".

Both downloads are on first use only; nothing here touches rengu's venv.

The backend is model-agnostic: :data:`GGUF_MODELS` maps a model id to its GGUF repo, vision
projector, weight quants and server tuning. ``toriigate-0.5`` is the caption stage's model;
the ``qwen3-vl-*`` entries serve the ``edit_caption`` stage, which sends several images per
request (controls, then the target) — :func:`server_budget` sizes the server context so
those requests fit a slot.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)

# Pinned llama.cpp release that supports ToriiGate's qwen3_5 vision arch. Bump deliberately.
LLAMACPP_RELEASE = "b9837"
_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{rel}/{asset}"


@dataclass(frozen=True)
class GGUFModelSpec:
    """One llama.cpp-servable VLM: where its files live and how to run its server.

    ``ctx_size``/``n_parallel`` are the fixed server tuning for single-image captioning;
    multi-image callers size the context per run with :func:`server_budget` instead.
    ``px_per_token`` is the image area (px) one LLM token covers — (patch × spatial merge)² —
    the input of that budget. ``tasks`` are the prep stages the model is offered in.
    """

    id: str
    repo: str
    mmproj: str
    quants: dict[str, str]
    default_quant: str
    ctx_size: int = 32768
    n_parallel: int = 16
    max_pixels: int = 1_000_000
    px_per_token: int = 32 * 32
    default_temperature: float = 0.7
    default_top_p: float = 0.8
    notes: str = ""
    tasks: tuple[str, ...] = ("caption",)


def _unsloth_qwen3_vl(size: str) -> dict[str, str]:
    stem = f"Qwen3-VL-{size}-Instruct"
    return {q: f"{stem}-{q}.gguf" for q in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")}


GGUF_MODELS: dict[str, GGUFModelSpec] = {
    "toriigate-0.5": GGUFModelSpec(
        id="toriigate-0.5",
        repo="DraconicDragon/ToriiGate-0.5-GGUF",
        mmproj="ToriiGate-0.5-fp16.mmproj.gguf",  # fp16 vision projector (small, keep quality)
        # Selectable weight quantizations, smallest/fastest -> largest/best. Q8_0 ≈ lossless.
        quants={
            "Q4_K_M": "ToriiGate-0.5-Q4_K_M.gguf",
            "Q5_K_M": "ToriiGate-0.5-Q5_K_M.gguf",
            "Q6_K": "ToriiGate-0.5-Q6_K.gguf",
            "Q8_0": "ToriiGate-0.5-Q8_0.gguf",
        },
        default_quant="Q8_0",
        # Validated server tuning (see the optimization notes in docs/user/dataset-prep.md):
        # 16 slots saturates the GPU; defaults beat -fa/-ub overrides; images capped to ~1 Mpx.
        ctx_size=32768,
        n_parallel=16,
        max_pixels=1_000_000,
        default_temperature=0.5,
        default_top_p=1.0,
        notes="Anime captioner (caption stage, engine = gguf).",
    ),
    # Qwen3-VL *Instruct* (not Thinking: no <think> preamble). Vision: 16 px patches with a 2x2
    # spatial merge -> one LLM token per 32x32 px. unsloth's repos carry the whole K-quant
    # ladder; the official Qwen/*-GGUF repos only ship Q4_K_M / Q8_0 / F16.
    "qwen3-vl-4b-instruct": GGUFModelSpec(
        id="qwen3-vl-4b-instruct",
        repo="unsloth/Qwen3-VL-4B-Instruct-GGUF",
        mmproj="mmproj-F16.gguf",
        quants=_unsloth_qwen3_vl("4B"),
        default_quant="Q8_0",  # 4.3 GB + 0.8 GB mmproj: fits an 8 GB card at the edit budget
        n_parallel=4,
        max_pixels=512 * 1024,
        notes="Qwen3-VL 4B Instruct (GGUF) — edit instructions; fits an 8 GB card. Default.",
        tasks=("edit_caption",),
    ),
    "qwen3-vl-8b-instruct": GGUFModelSpec(
        id="qwen3-vl-8b-instruct",
        repo="unsloth/Qwen3-VL-8B-Instruct-GGUF",
        mmproj="mmproj-F16.gguf",
        quants=_unsloth_qwen3_vl("8B"),
        default_quant="Q4_K_M",  # 5.0 GB + 1.2 GB mmproj; Q8_0 (8.7 GB) needs a 12 GB+ card
        n_parallel=4,
        max_pixels=512 * 1024,
        notes="Qwen3-VL 8B Instruct (GGUF) — stronger edit instructions; Q4_K_M ~7 GB, Q8_0 needs 12 GB+.",
        tasks=("edit_caption",),
    ),
}

# ToriiGate aliases: the caption stage's gguf engine (and its tests) read these names.
_TORII = GGUF_MODELS["toriigate-0.5"]
GGUF_REPO = _TORII.repo
MMPROJ_FILE = _TORII.mmproj
GGUF_QUANTS: dict[str, str] = _TORII.quants
DEFAULT_QUANT = _TORII.default_quant
N_PARALLEL = _TORII.n_parallel
CTX_SIZE = _TORII.ctx_size
MAX_PIXELS = _TORII.max_pixels


def gguf_models(task: str | None = None) -> dict[str, GGUFModelSpec]:
    """Registered GGUF models, optionally only those offered for ``task`` (a prep stage)."""
    return {k: v for k, v in GGUF_MODELS.items() if task is None or task in v.tasks}


def get_gguf_model(model: str) -> GGUFModelSpec:
    spec = GGUF_MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unknown GGUF model {model!r}. Known: {list(GGUF_MODELS)}")
    return spec


def resolve_quant(spec: GGUFModelSpec, quant: str | None) -> str:
    """The quant to download: ``quant`` when the model ships it, else the model's default."""
    if quant and quant in spec.quants:
        return quant
    if quant:
        logger.warning("%s has no GGUF quant %r; using %s", spec.id, quant, spec.default_quant)
    return spec.default_quant


def image_tokens(max_pixels: int, px_per_token: int) -> int:
    """Upper bound of the LLM tokens one image of <= ``max_pixels`` costs.

    The vision preprocessor rounds each side UP to its patch grid, so a resized image can land
    slightly above ``max_pixels``; the 10 % + 8 margin covers that and the vision marker tokens.
    """
    return int(max_pixels * 1.1 / px_per_token) + 8


def server_budget(
    spec: GGUFModelSpec,
    *,
    n_images: int,
    max_pixels: int,
    max_new_tokens: int,
    n_parallel: int | None = None,
    prompt_tokens: int = 512,
) -> tuple[int, int]:
    """``(ctx_size, n_parallel)`` for a server whose requests carry up to ``n_images`` images.

    llama-server splits ``-c`` evenly across its ``--parallel`` slots and a request larger than
    its slot fails. ToriiGate's fixed ``-c 32768 --parallel 16`` leaves 2048 tokens per slot:
    enough for one ~1 Mpx image, not for a control + target pair. So the slot is sized from the
    worst request (every image at ``max_pixels``, plus prompt and output) and the context is
    ``slot × parallel``. The trade-off is VRAM: the KV cache grows linearly with the context, so
    more images per row, more pixels per image or more slots all cost memory — lower
    ``max_pixels`` (coarser view of the edit) or ``n_parallel`` (less throughput) to fit.
    """
    parallel = max(1, int(n_parallel or spec.n_parallel))
    slot = n_images * image_tokens(max_pixels, spec.px_per_token) + prompt_tokens + max_new_tokens
    slot = -(-slot // 256) * 256  # round up to a multiple of 256
    return slot * parallel, parallel


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    d = Path(base) / "rengu-flow" / "llamacpp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _release_asset() -> str:
    """Pick the Vulkan release asset for this platform (GPU without a CUDA match)."""
    import sys

    from rengu_flow.platform_compat import PLATFORM

    if PLATFORM.is_windows:
        return f"llama-{LLAMACPP_RELEASE}-bin-win-vulkan-x64.zip"
    if sys.platform == "darwin":  # Platform has no is_macos flag; detect macOS directly
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


def ensure_gguf(quant: str, model: str = "toriigate-0.5") -> tuple[Path, Path]:
    """Download ``model``'s weight GGUF (``quant``, else its default) + its mmproj via the HF
    cache; return both paths."""
    from huggingface_hub import hf_hub_download

    spec = get_gguf_model(model)
    fname = spec.quants[resolve_quant(spec, quant)]
    gguf = Path(hf_hub_download(repo_id=spec.repo, filename=fname))
    mmproj = Path(hf_hub_download(repo_id=spec.repo, filename=spec.mmproj))
    return gguf, mmproj


def _server_env(bin_dir: Path) -> dict:
    env = dict(os.environ)
    # The release ships its shared libs (libggml*, vulkan backend) next to the binary.
    env["LD_LIBRARY_PATH"] = f"{bin_dir}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _start_server(
    bin_dir: Path, gguf: Path, mmproj: Path, port: int,
    *, ctx_size: int = CTX_SIZE, n_parallel: int = N_PARALLEL,
):
    exe = bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
    cmd = [
        str(exe), "-m", str(gguf), "--mmproj", str(mmproj),
        "-ngl", "99", "-c", str(ctx_size), "--parallel", str(n_parallel),
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


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # 0 = OS assigns a free ephemeral port (never hard-code 8080)
        return s.getsockname()[1]


def _stop(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def llama_server(
    bin_dir: Path, gguf: Path, mmproj: Path,
    *, ctx_size: int = CTX_SIZE, n_parallel: int = N_PARALLEL,
) -> Iterator[int]:
    """Run ``llama-server`` for the ``with`` block; yields its port and always stops it.

    Starts on a free port; if it loses the race for that port (closed before the server
    binds), retries on a fresh one — no hard-coded port a dev server might already hold.
    """
    proc = port = None
    for attempt in range(3):
        port = _free_port()
        proc = _start_server(bin_dir, gguf, mmproj, port, ctx_size=ctx_size, n_parallel=n_parallel)
        try:
            _wait_health(port, proc, timeout=180.0 if attempt == 0 else 30.0)
            break
        except Exception as exc:  # noqa: BLE001
            _stop(proc)
            if attempt == 2:
                raise
            logger.warning("llama-server start failed on port %d (%s); retrying", port, exc)
    try:
        yield port
    finally:
        _stop(proc)


def _encode_image(path: Path, max_pixels: int = MAX_PIXELS) -> str:
    """Resize to <= ``max_pixels`` (ToriiGate: its training res) and return base64 JPEG.

    The cap bounds the vision tokens, so a request can't overflow a server slot.
    """
    import base64
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    if im.width * im.height > max_pixels:
        s = (max_pixels / (im.width * im.height)) ** 0.5
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def image_part(b64: str) -> dict:
    """One OpenAI-style image content part for a base64 JPEG."""
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _request_caption(port: int, b64: str, prompt: str, config) -> str:
    """Single-image caption request with ToriiGate's sampling defaults."""
    return request_chat(
        port, [image_part(b64), {"type": "text", "text": prompt}], config,
        default_temperature=0.5, default_top_p=1.0,
    )


def request_chat(
    port: int, content: list[dict], config,
    *, default_temperature: float, default_top_p: float,
) -> str:
    """Send one user turn of ``content`` parts (text and images, in order); return the reply.

    ``config`` supplies ``temperature`` / ``top_p`` (``None`` = the given defaults) and
    ``max_new_tokens``.
    """
    body = {
        "messages": [{"role": "user", "content": content}],
        "temperature": config.temperature if config.temperature is not None else default_temperature,
        "top_p": config.top_p if config.top_p is not None else default_top_p,
        "max_tokens": config.max_new_tokens,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

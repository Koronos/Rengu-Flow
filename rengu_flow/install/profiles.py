"""Optional dependency profiles: extras, import-checks, git requirements, uv argv builders."""

from __future__ import annotations

# Short CLI name -> pyproject optional-dependencies extra (None = base uv sync only).
PROFILE_EXTRAS: dict[str, str | None] = {
    "base": None,
    "ui": "ui",
    "cosmos": "cosmos_predict2",
    "cosmos_predict2": "cosmos_predict2",
    "optim": "optim",
    "lycoris": "lycoris",
    "dev": "dev",
    # Git-backed extra: the `koptim` package is declared in pyproject with a [tool.uv.sources]
    # git source, so uv installs and version-manages it like any other extra (lockfile-pinned).
    "koptim": "koptim",
}

PROFILE_LABELS: dict[str, str] = {
    "base": "Base (training core)",
    "ui": "Web UI",
    "cosmos": "Cosmos Predict2",
    "cosmos_predict2": "Cosmos Predict2",
    "optim": "Optimizers (bitsandbytes, torchao, …)",
    "lycoris": "LyCORIS adapters",
    "dev": "Development (pytest, httpx)",
    "koptim": "K-Optimizers (Adafusion, Muon, AdaMuon)",
}

PROFILE_DESCRIPTIONS: dict[str, str] = {
    "base": "Core training dependencies from uv.lock.",
    "ui": "FastAPI control plane and TensorBoard integration.",
    "cosmos": "transformers, accelerate, torchvision, einops for Cosmos training.",
    "cosmos_predict2": "Same as cosmos.",
    "optim": "Optional optimizer backends (bitsandbytes, pytorch-optimizer, …).",
    "lycoris": "LyCORIS-style adapter backend.",
    "dev": "pytest and httpx for development.",
    "koptim": "Memory-efficient Adafusion / Muon / AdaMuon optimizers from github.com/Koronos/K-Optimizers.",
}

ALL_PROFILE_NAMES = ("base", "ui", "cosmos", "optim", "lycoris", "dev", "koptim")

# Profile -> modules that must import for the profile to count as installed. Used by the manager
# to decide whether anything needs installing (on-demand) and to verify success afterwards.
PROFILE_IMPORT_CHECKS: dict[str, tuple[str, ...]] = {
    "ui": ("fastapi", "uvicorn"),
    "cosmos": ("transformers", "einops"),
    "cosmos_predict2": ("transformers", "einops"),
    "lycoris": ("lycoris",),
    "optim": ("bitsandbytes",),
    "dev": ("pytest",),
    "koptim": ("koptim",),
}

# Profile -> pip/git requirement specs that uv cannot install via pyproject extras. Installed
# additively with ``uv pip install`` when the profile's modules are still missing after the
# regular sync. Prefer a pyproject extra + [tool.uv.sources] git source (like ``koptim``) so the
# package is lockfile-pinned and upgradable via PROFILE_GIT_PACKAGES below. Only use this escape
# hatch for specs uv's project workflow genuinely can't express; register them as:
#     "myoptim": ["git+https://github.com/acme/cool-optimizer@v1.2.0"],
# and add the importable module name to PROFILE_IMPORT_CHECKS above so detection works.
PROFILE_GIT_REQUIREMENTS: dict[str, list[str]] = {}


def normalize_profiles(names: list[str]) -> list[str]:
    """Expand ``all``, dedupe, preserve order."""
    out: list[str] = []
    for name in names:
        key = name.strip().lower()
        if not key:
            continue
        if key == "all":
            for p in ALL_PROFILE_NAMES:
                if p not in out:
                    out.append(p)
            continue
        if key not in PROFILE_EXTRAS:
            raise ValueError(
                f"Unknown profile {name!r}; choose from: {', '.join(sorted(PROFILE_EXTRAS))}, all"
            )
        if key not in out:
            out.append(key)
    return out or ["base"]


def uv_sync_argv(profiles: list[str]) -> list[str]:
    """Build argv for ``uv sync`` with optional extras.

    Uses ``--inexact`` so the sync is **additive**: packages already in the environment that are
    not part of the selected resolution (other extras, user-installed custom optimizers/schedulers,
    git packages) are preserved instead of being removed.

    Git-sourced extras (e.g. ``koptim``) are pinned to an exact commit in [tool.uv.sources], so
    selecting the extra installs exactly that revision; the package only changes when the pin is
    bumped in pyproject — uv re-locks and applies it on the next sync.
    """
    normalized = normalize_profiles(profiles)
    cmd = ["uv", "sync", "--inexact"]
    extras: set[str] = set()
    for p in normalized:
        extra = PROFILE_EXTRAS.get(p)
        if extra:
            extras.add(extra)
    for extra in sorted(extras):
        cmd.extend(["--extra", extra])
    return cmd


def rengu_init_command(profiles: list[str]) -> str:
    normalized = normalize_profiles(profiles)
    if normalized == ["base"]:
        return "rengu init"
    return "rengu init " + " ".join(normalized)


def profile_metadata() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in ALL_PROFILE_NAMES:
        rows.append(
            {
                "id": key,
                "label": PROFILE_LABELS.get(key, key),
                "description": PROFILE_DESCRIPTIONS.get(key, ""),
                "command": rengu_init_command([key]),
            }
        )
    return rows

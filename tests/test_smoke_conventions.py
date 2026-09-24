"""Guards for the smoke convention (docs/developer/smoke-tests.md): no personal/absolute
paths in smoke files, model paths only via RENGU_*_PATH documented in .env.example."""

import re
from pathlib import Path

import pytest
import toml

from rengu_flow.config.local_env import _MODEL_PATH_ENV

pytestmark = pytest.mark.no_ui_db

REPO = Path(__file__).resolve().parents[1]
SMOKE_FIXTURES = sorted((REPO / "tests" / "fixtures" / "smoke").glob("*.toml"))
SMOKE_FILES = sorted(
    {
        *REPO.glob("scripts/*smoke*"),
        *REPO.glob("scripts/run_*.sh"),
        REPO / "scripts" / "lib" / "smoke_common.sh",
        REPO / ".env.example",
        *SMOKE_FIXTURES,
    }
)
# Home dirs (Windows, Git Bash, WSL mounts, Linux, macOS), WSL UNC shares, AppData.
PERSONAL_PATH = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]|/c/Users/|/mnt/[a-z]/Users/|/home/[A-Za-z]|/Users/[A-Za-z]"
    r"|wsl\.localhost|wsl\$|AppData",
    re.IGNORECASE,
)
ENV_VAR = re.compile(r"\bRENGU_[A-Z0-9_]+_PATH\b")


def _documented_env_vars() -> set[str]:
    return set(ENV_VAR.findall((REPO / ".env.example").read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", SMOKE_FILES, ids=lambda p: p.relative_to(REPO).as_posix())
def test_smoke_file_has_no_personal_paths(path):
    hits = PERSONAL_PATH.findall(path.read_text(encoding="utf-8"))
    assert not hits, f"{path.name}: absolute/personal path(s) {hits} — use .env + local_env"


@pytest.mark.parametrize("path", SMOKE_FIXTURES, ids=lambda p: p.name)
def test_smoke_fixture_has_no_model_paths(path):
    model = toml.load(path).get("model", {})
    paths = [k for k in model if k.endswith("_path")]
    assert not paths, f"{path.name}: [model] {paths} — set RENGU_*_PATH in .env instead"


def test_every_model_path_env_var_is_in_env_example():
    used = {v for mapping in _MODEL_PATH_ENV.values() for v in mapping.values()}
    for root in ("rengu_flow", "scripts", "tests/fixtures/smoke"):
        for f in (REPO / root).rglob("*"):
            if f.is_file() and f.suffix in {".py", ".sh", ".toml"}:
                used |= set(ENV_VAR.findall(f.read_text(encoding="utf-8", errors="ignore")))
    missing = sorted(used - _documented_env_vars())
    assert not missing, f"document in .env.example: {missing}"

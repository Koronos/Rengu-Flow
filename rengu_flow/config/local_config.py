"""Load repo-root ``rengu.local.toml`` for CLI, UI, and training launcher defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import toml

LOCAL_CONFIG_FILENAME = "rengu.local.toml"
LOCAL_CONFIG_EXAMPLE = "rengu.local.toml.example"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / LOCAL_CONFIG_FILENAME


def local_config_example_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / LOCAL_CONFIG_EXAMPLE


@dataclass
class UiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    # When true, bind the UI to all interfaces (0.0.0.0) so other devices on your local network
    # can reach it, instead of localhost only. Pair with `token` — anyone on the network can then
    # drive training. Overrides `host` when set.
    public: bool = False
    data_dir: str = "data"
    token: str | None = None


# Hosts that mean "bind every interface" (reachable from the local network), used for the public
# exposure check.
_ALL_INTERFACES_HOSTS = ("0.0.0.0", "::", "*")


def public_bind_warning(host: str, token: str | None) -> str | None:
    """Warning text when the UI is exposed to the network without a token, else None."""
    if host in _ALL_INTERFACES_HOSTS and not token:
        return (
            "UI is reachable from your local network (ui.public = true) but no ui.token is set — "
            "anyone on the network can start/stop training, edit configs, and browse dataset "
            "paths. Set [ui] token in rengu.local.toml to require a token."
        )
    return None


@dataclass
class MaintenanceConfig:
    enabled: bool = False
    allow_pip: bool = False


@dataclass
class TrainingConfig:
    num_gpus: int = 1
    master_port: int = 29500
    extra_args: str = ""
    # Training engine backend: "" (auto: deepspeed on Linux, accelerate on Windows),
    # "deepspeed", or "accelerate". See rengu_flow.engine.
    engine: str = ""
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolboxConfig:
    enabled: bool = False


@dataclass
class LocalConfig:
    root: Path
    ui: UiConfig = field(default_factory=UiConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)

    def ui_data_dir(self) -> Path:
        p = Path(self.ui.data_dir)
        if p.is_absolute():
            return p.resolve()
        return (self.root / p).resolve()

    def ui_bind_host(self) -> str:
        """Host the server binds to: all interfaces when `public`, else the configured host."""
        return "0.0.0.0" if self.ui.public else self.ui.host


_loaded: LocalConfig | None = None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_training_env(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        out[key.strip()] = str(value)
    return out


def parse_local_config_dict(data: dict[str, Any], *, root: Path) -> LocalConfig:
    ui_raw = data.get("ui") if isinstance(data.get("ui"), dict) else {}
    maint_raw = data.get("maintenance") if isinstance(data.get("maintenance"), dict) else {}
    train_raw = data.get("training") if isinstance(data.get("training"), dict) else {}
    env_raw = train_raw.get("env") if isinstance(train_raw.get("env"), dict) else {}

    raw_data_dir = str(ui_raw.get("data_dir", "data")).strip() or "data"
    ui = UiConfig(
        host=str(ui_raw.get("host", "127.0.0.1")),
        port=int(ui_raw.get("port", 8765)),
        public=_boolish(ui_raw.get("public", False)),
        data_dir=raw_data_dir,
        token=str(ui_raw["token"]).strip() if ui_raw.get("token") else None,
    )
    maintenance = MaintenanceConfig(
        enabled=_boolish(maint_raw.get("enabled", False)),
        allow_pip=_boolish(maint_raw.get("allow_pip", False)),
    )
    training = TrainingConfig(
        num_gpus=int(train_raw.get("num_gpus", 1)),
        master_port=int(train_raw.get("master_port", 29500)),
        extra_args=str(train_raw.get("extra_args", "")).strip(),
        engine=str(train_raw.get("engine", "")).strip().lower(),
        env=_parse_training_env(env_raw),
    )
    toolbox_raw = data.get("toolbox") if isinstance(data.get("toolbox"), dict) else {}
    toolbox = ToolboxConfig(enabled=_boolish(toolbox_raw.get("enabled", False)))
    return LocalConfig(root=root, ui=ui, maintenance=maintenance, training=training, toolbox=toolbox)


def load_local_config(path: Path | None = None, *, root: Path | None = None) -> LocalConfig | None:
    """Parse ``rengu.local.toml`` if present. Returns None when the file is missing."""
    global _loaded
    r = root or repo_root()
    cfg_path = path if path is not None else local_config_path(r)
    if not cfg_path.is_file():
        _loaded = None
        return None
    data = toml.load(cfg_path)
    if not isinstance(data, dict):
        _loaded = None
        return None
    _loaded = parse_local_config_dict(data, root=r)
    return _loaded


def get_local_config() -> LocalConfig | None:
    return _loaded


def toolbox_enabled() -> bool:
    """True when Toolbox tool *execution* is allowed (authoring is always allowed)."""
    return ensure_local_config_loaded().toolbox.enabled


def default_local_config(root: Path | None = None) -> LocalConfig:
    return LocalConfig(root=root or repo_root())


def ensure_local_config_loaded() -> LocalConfig:
    cached = get_local_config()
    if cached is not None:
        return cached
    cfg = load_local_config()
    if cfg is not None:
        return cfg
    default = default_local_config()
    global _loaded
    _loaded = default
    return default


def apply_local_config_to_environ(cfg: LocalConfig | None = None) -> None:
    """Apply UI and maintenance settings to ``os.environ`` (setdefault)."""
    c = cfg if cfg is not None else ensure_local_config_loaded()
    os.environ["RENGU_FLOW_UI_HOST"] = c.ui_bind_host()
    os.environ["RENGU_FLOW_UI_PORT"] = str(c.ui.port)
    os.environ["RENGU_FLOW_UI_DATA"] = str(c.ui_data_dir())
    if c.ui.token:
        os.environ.setdefault("RENGU_FLOW_UI_TOKEN", c.ui.token)
    os.environ.setdefault("RENGUFLOW_MAINTENANCE", "1" if c.maintenance.enabled else "0")
    os.environ.setdefault(
        "RENGUFLOW_MAINTENANCE_ALLOW_PIP",
        "1" if c.maintenance.allow_pip else "0",
    )


def init_local_config_file(*, root: Path | None = None, force: bool = False) -> Path:
    """Copy example to ``rengu.local.toml`` when missing (or when ``force``)."""
    r = root or repo_root()
    dest = local_config_path(r)
    if dest.is_file() and not force:
        return dest
    example = local_config_example_path(r)
    if not example.is_file():
        raise FileNotFoundError(f"Missing {example}")
    dest.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def ensure_local_config_file(*, root: Path | None = None, quiet: bool = False) -> Path | None:
    """Create ``rengu.local.toml`` from the example when it is missing.

    For users who skipped ``rengu init``: the UI/CLI calls this on startup so the config that
    holds the UI port and training defaults always exists. Idempotent — an existing file is left
    untouched — and it never raises: a config that cannot be written must not block startup,
    since the loader falls back to defaults anyway.
    """
    r = root or repo_root()
    dest = local_config_path(r)
    if dest.is_file():
        return dest
    example = local_config_example_path(r)
    if not example.is_file():
        if not quiet:
            print(f"==> {LOCAL_CONFIG_EXAMPLE} not found; continuing with built-in defaults.")
        return None
    try:
        dest.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError as e:
        if not quiet:
            print(f"==> Could not write {dest.name} ({e}); continuing with built-in defaults.")
        return None
    if not quiet:
        print(f"==> Generated {dest.name} (UI port + training defaults). Edit it to customize.")
    return dest


def ensure_ui_data_dir(cfg: LocalConfig | None = None) -> Path:
    c = cfg if cfg is not None else ensure_local_config_loaded()
    d = c.ui_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

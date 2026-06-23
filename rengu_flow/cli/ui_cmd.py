"""``rengu ui`` subcommands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which

from rengu_flow.config.local_config import ensure_local_config_loaded, repo_root
from rengu_flow.cli.project_venv import reexec_cli
from rengu_flow.install import ensure_ui_dependencies, self_heal
from rengu_flow.platform_compat import PLATFORM, free_port_owned_by


def add_parser(sub: argparse._SubParsersAction) -> None:
    ui = sub.add_parser("ui", help="Web UI control plane (defaults to `start`)")
    # Bare `rengu ui` starts the UI: the subcommand is optional and defaults to `start`.
    ui.set_defaults(ui_command=None)
    ui_sub = ui.add_subparsers(dest="ui_command")

    start = ui_sub.add_parser("start", help="Install UI extra, build web dist, run server")
    start.add_argument("--no-open", action="store_true")
    start.add_argument("--rebuild-web", action="store_true")
    start.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip uv sync (launchers use this after uv sync --extra ui)",
    )

    dev = ui_sub.add_parser("dev", help="API reload + Vite dev server")
    dev.add_argument("--no-open", action="store_true")
    dev.add_argument("--dev-port", type=int, default=5173)
    dev.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip uv sync (launchers use this after uv sync --extra ui)",
    )
    dev.add_argument(
        "--no-reload-api",
        action="store_true",
        help="Disable API auto-reload on .py changes (faster startup, no hot-reload)",
    )

    serve = ui_sub.add_parser("serve", help="Run API server only")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")

    ui_sub.add_parser("build", help="Build ui/web frontend (pnpm)")
    ui_sub.add_parser("reset-db", help="Reset UI SQLite database")


# Frontend package manager: prefer pnpm, fall back to npm. The registry override bypasses a private
# (CodeArtifact) default that breaks public-package installs; overridable via RENGU_UI_NPM_REGISTRY.
PNPM_REGISTRY = os.environ.get("RENGU_UI_NPM_REGISTRY", "https://registry.npmjs.org/")


def _real_node_dir() -> str | None:
    """Directory of the *actual* node binary, resolved past wrapper shims. Volta's shim
    (``C:\\Program Files\\Volta\\node.exe``) runs fine when invoked directly but breaks in the nested
    build scripts (vue-tsc → node, vite → node) when ``VOLTA_HOME`` is unset, and its path has a space
    that trips some shells. Asking node for ``process.execPath`` returns the real, space-free binary
    (e.g. ``…\\Volta\\tools\\image\\node\\<ver>``); putting that dir first on PATH fixes the nesting."""
    if not which("node"):
        return None
    try:
        out = subprocess.run(
            ["node", "-e", "process.stdout.write(process.execPath)"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return str(Path(path).parent) if out.returncode == 0 and path else None


def _build_env() -> dict[str, str]:
    """Env for the frontend build with the real node dir prepended to PATH (see _real_node_dir)."""
    env = os.environ.copy()
    node_dir = _real_node_dir()
    if node_dir:
        env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")
    return env


def _pick_pm() -> str | None:
    """pnpm if present, else npm, else None."""
    if which("pnpm"):
        return "pnpm"
    if which("npm"):
        return "npm"
    return None


def _ensure_node() -> None:
    """Ensure Node.js + a JS package manager are usable for the frontend build, with clear errors."""
    if not which("node"):
        nvm_dir = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm"))
        if (nvm_dir / "nvm.sh").is_file():  # nvm must be sourced in shell; try common node path
            versions = sorted((nvm_dir / "versions" / "node").glob("*/bin/node"))
            if versions:
                os.environ["PATH"] = f"{versions[-1].parent}{os.pathsep}{os.environ.get('PATH', '')}"
    if not which("node"):
        raise SystemExit(
            "rengu: Node.js not found — the web UI needs it to build the frontend. Install it "
            "(https://nodejs.org, or via Volta / nvm), then retry. To skip the build use a prebuilt "
            "ui/web/dist, or run the API only: `rengu ui serve`."
        )
    if _pick_pm() is None:
        raise SystemExit(
            "rengu: no JS package manager found. Install pnpm (recommended: `corepack enable pnpm` "
            "or `volta install pnpm`) or npm, then retry."
        )


def _web_dir(root: Path) -> Path:
    return root / "ui" / "web"


def _pm_install_cmd(pm: str) -> list[str]:
    """Install argv for *pm*. pnpm needs a hoisted (flat) layout so the UI's direct imports of
    transitive @codemirror sub-packages resolve; npm hoists by default. Both bypass the private default
    registry."""
    if pm == "pnpm":
        return ["pnpm", "install", "--config.node-linker=hoisted", f"--registry={PNPM_REGISTRY}"]
    return ["npm", "install", f"--registry={PNPM_REGISTRY}"]


def _build_web(root: Path, *, force: bool = False) -> None:
    web = _web_dir(root)
    dist = web / "dist" / "index.html"
    if dist.is_file() and not force:
        return
    _ensure_node()
    pm = _pick_pm()
    env = _build_env()
    print(f"==> Building web frontend ({pm})...")
    subprocess.run(_pm_install_cmd(pm), cwd=str(web), check=True, env=env)
    subprocess.run([pm, "run", "build"], cwd=str(web), check=True, env=env)
    if not dist.is_file():
        raise SystemExit(f"rengu: build finished but {dist} is missing")


def _browser_open(url: str) -> None:
    if which("xdg-open"):
        subprocess.run(["xdg-open", url], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"Open in browser: {url}")


def _wait_health(
    url: str,
    token: str | None,
    timeout: float = 90.0,
    *,
    label: str = "server",
) -> bool:
    # Pure stdlib HTTP poll — NOT curl. The old `curl -sf -o /dev/null` returned non-zero on Windows
    # (its native curl can't write the POSIX path /dev/null -> CURLE_WRITE_ERROR) *despite* a 200, so
    # the wait looped forever printing "Waiting for server" while the server was already healthy.
    import urllib.error
    import urllib.request

    headers = {"X-Rengu-Flow-Token": token} if token else {}
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=2
            ) as resp:
                if resp.status < 400:
                    return True
        except (urllib.error.URLError, OSError):
            pass  # not up yet (connection refused / timeout) or 4xx (e.g. token) -> keep waiting
        attempt += 1
        if attempt == 1 or attempt % 10 == 0:
            print(f"==> Waiting for {label} ({url})...", flush=True)
        time.sleep(0.4)
    return False


def _uvicorn_reload_excludes() -> list[str]:
    """Ignore noisy paths so the dev reloader does not restart in a loop."""
    return [
        "**/.git",
        "**/.venv",
        "**/data",
        "**/output",
        "**/ui/web/node_modules",
        "**/ui/web/dist",
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.db",
        "**/*.db-journal",
        "**/*.log",
        "**/tmp",
        "**/.pytest_cache",
    ]


def _warn_public(host: str, token: str | None) -> None:
    """Print a one-line warning when the UI is exposed to the network without a token."""
    from rengu_flow.config.local_config import public_bind_warning

    msg = public_bind_warning(host, token)
    if msg:
        print(f"==> WARNING: {msg}", flush=True)


def _serve(host: str, port: int, *, reload: bool = False) -> None:
    import uvicorn

    if reload:
        from rengu_flow_ui import settings as ui_settings

        pkg_root = Path(ui_settings.__file__).resolve().parent
        uvicorn.run(
            "rengu_flow_ui.app:create_app",
            factory=True,
            host=host,
            port=port,
            log_level="info",
            reload=True,
            reload_dirs=[str(pkg_root)],
            reload_excludes=_uvicorn_reload_excludes(),
            reload_delay=0.5,
        )
    else:
        from rengu_flow_ui.app import create_app

        uvicorn.run(create_app(), host=host, port=port, log_level="info")


def run(args: argparse.Namespace) -> None:
    root = repo_root()
    reexec_cli()
    cfg = ensure_local_config_loaded()
    # Default to `start` so `rengu ui` (no subcommand) launches the control panel.
    cmd = args.ui_command or "start"

    # Check the DB schema here, in the terminal-attached process, before building the web
    # dist, spawning the API subprocess, or constructing the app (which runs init_db). An
    # incompatible schema must get the user's consent to wipe-and-recreate first — we don't
    # want the prompt buried in a subprocess log, or skipped entirely on the start/serve paths.
    if cmd in ("start", "dev", "serve"):
        from rengu_flow_ui.schema_guard import ensure_schema_compatible

        ensure_schema_compatible()

    if cmd == "reset-db":
        from rengu_flow_ui.db import reset_ui_database

        path = reset_ui_database()
        print(f"Reset UI database: {path}")
        return

    if cmd == "build":
        _build_web(root, force=True)
        print(f"==> Web build OK: {_web_dir(root) / 'dist'}")
        return

    if cmd == "start":
        # Flags only exist when the `start` subparser ran; bare `rengu ui` uses defaults.
        skip_sync = getattr(args, "skip_sync", False)
        rebuild_web = getattr(args, "rebuild_web", False)
        no_open = getattr(args, "no_open", False)
        if not skip_sync:
            self_heal()
            ensure_ui_dependencies()
            reexec_cli()
        _build_web(root, force=rebuild_web)
        host = cfg.ui_bind_host()
        port = cfg.ui.port
        free_port_owned_by(port, patterns=("rengu", "uvicorn", "rengu-flow-ui"))
        _warn_public(host, cfg.ui.token)
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::", "*") else host
        url = f"http://{browser_host}:{port}/"
        health = f"{url.rstrip('/')}/api/v1/health"
        if not no_open:

            def _open_when_ready() -> None:
                if _wait_health(health, cfg.ui.token):
                    _browser_open(url)

            threading.Thread(target=_open_when_ready, daemon=True).start()
        print(f"==> Rengu Flow UI: {url}")
        if PLATFORM.is_windows:
            # Windows only: run the server as a separate python process so the long-lived server is
            # python.exe — never the rengu.exe console-script. The OS forbids replacing a running
            # rengu.exe, so an in-process server would lock it and make every later `uv sync` (e.g.
            # installing the prep extra) fail with WinError 32. POSIX can replace a running exe, so
            # it keeps the simpler in-process server below.
            serve_cmd = [sys.executable, "-m", "rengu_flow_ui", "serve", "--host", host, "--port", str(port)]
            serve_env = os.environ.copy()
            serve_env.setdefault("PYTHONUNBUFFERED", "1")
            proc = subprocess.Popen(serve_cmd, cwd=str(root), env=serve_env)
            try:
                raise SystemExit(proc.wait())
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise SystemExit(0)
        _serve(host, port, reload=False)
        return

    if cmd == "dev":
        reload_api = not args.no_reload_api
        if not args.skip_sync:
            self_heal()
            ensure_ui_dependencies()
            reexec_cli()
        _ensure_node()
        dev_pm = _pick_pm()
        web = _web_dir(root)
        if not (web / "node_modules").is_dir():
            install_cmd = _pm_install_cmd(dev_pm)
            print(f"==> {' '.join(install_cmd)} (ui/web)...", flush=True)
            subprocess.run(install_cmd, cwd=str(web), check=True, env=_build_env())
        api_port = cfg.ui.port
        dev_port = args.dev_port
        free_port_owned_by(
            api_port,
            patterns=("rengu", "uvicorn", "rengu-flow-ui", "rengu_flow_ui", "multiprocessing"),
        )
        free_port_owned_by(dev_port, patterns=("vite", "node"))
        health = f"http://127.0.0.1:{api_port}/api/v1/health"
        api_cmd = [
            sys.executable,
            "-m",
            "rengu_flow_ui",
            "serve",
            "--host",
            cfg.ui.host,
            "--port",
            str(api_port),
        ]
        api_log = cfg.ui_data_dir() / "dev-api.log"
        api_log.parent.mkdir(parents=True, exist_ok=True)
        api_log.write_text("", encoding="utf-8")
        if reload_api:
            api_cmd.append("--reload")
            print(
                f"==> Starting API with reload on http://{cfg.ui.host}:{api_port}/ "
                "(first start can take ~60–90s on a cold import; log: "
                f"{api_log})...",
                flush=True,
            )
        else:
            print(
                f"==> Starting API on http://{cfg.ui.host}:{api_port}/ "
                f"(log: {api_log})...",
                flush=True,
            )
        api_env = os.environ.copy()
        api_env.setdefault("PYTHONUNBUFFERED", "1")
        api_log_fh = api_log.open("a", encoding="utf-8")
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=str(root),
            env=api_env,
            stdout=api_log_fh,
            stderr=subprocess.STDOUT,
        )
        health_timeout = 180.0 if reload_api else 90.0
        if not _wait_health(health, cfg.ui.token, timeout=health_timeout, label="API"):
            api_poll = api_proc.poll()
            if api_poll is not None:
                print(f"==> API process exited with code {api_poll}", flush=True)
            print(f"==> API log (last 40 lines): {api_log}", flush=True)
            try:
                lines = api_log.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in lines[-40:]:
                    print(line)
            except OSError:
                pass
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_proc.kill()
            api_log_fh.close()
            raise SystemExit(
                "rengu: API did not become healthy for dev mode. "
                f"See {api_log} or: rengu ui serve --reload"
            )
        dev_url = f"http://127.0.0.1:{dev_port}/"
        print(f"==> Dev UI: {dev_url}", flush=True)
        print(f"==> API (proxied): http://127.0.0.1:{api_port}/api/v1/", flush=True)
        npm_env = _build_env()  # real node dir on PATH for nested vite -> node
        npm_env["RENGU_FLOW_UI_PORT"] = str(api_port)
        npm_env["RENGU_FLOW_UI_DEV_PORT"] = str(dev_port)
        npm_cmd = [dev_pm, "run", "dev"]
        if not args.no_open:
            npm_cmd.extend(["--", "--open"])
        try:
            raise SystemExit(subprocess.run(npm_cmd, cwd=str(web), env=npm_env).returncode)
        finally:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api_proc.kill()

    if cmd == "serve":
        host = args.host or cfg.ui_bind_host()
        port = args.port if args.port is not None else cfg.ui.port
        _warn_public(host, cfg.ui.token)
        _serve(host, port, reload=args.reload)
        return

    raise SystemExit(f"unknown ui command: {cmd}")

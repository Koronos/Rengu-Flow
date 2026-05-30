"""``rengu ui`` subcommands."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from shutil import which

from rengu_flow.config.local_config import ensure_local_config_loaded, repo_root
from rengu_flow.cli.project_venv import reexec_cli
from rengu_flow.install import ensure_ui_dependencies, self_heal


def add_parser(sub: argparse._SubParsersAction) -> None:
    ui = sub.add_parser("ui", help="Web UI control plane")
    ui_sub = ui.add_subparsers(dest="ui_command", required=True)

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

    ui_sub.add_parser("build", help="Build ui/web frontend (npm run build)")
    ui_sub.add_parser("reset-db", help="Reset UI SQLite database")


def _ensure_node() -> None:
    if which("npm"):
        return
    nvm_dir = Path(os.environ.get("NVM_DIR", Path.home() / ".nvm"))
    nvm_sh = nvm_dir / "nvm.sh"
    if nvm_sh.is_file():
        # nvm must be sourced in shell; try common node path
        versions = sorted((nvm_dir / "versions" / "node").glob("*/bin/npm"))
        if versions:
            os.environ["PATH"] = f"{versions[-1].parent}:{os.environ.get('PATH', '')}"
    if not which("npm"):
        raise SystemExit(
            "rengu: npm not found. Install Node.js or nvm, then run: cd ui/web && npm ci && npm run build"
        )


def _web_dir(root: Path) -> Path:
    return root / "ui" / "web"


def _build_web(root: Path, *, force: bool = False) -> None:
    web = _web_dir(root)
    dist = web / "dist" / "index.html"
    if dist.is_file() and not force:
        return
    _ensure_node()
    print("==> Building web frontend...")
    lock = web / "package-lock.json"
    install_cmd = ["npm", "ci"] if lock.is_file() else ["npm", "install"]
    subprocess.run(install_cmd, cwd=str(web), check=True)
    subprocess.run(["npm", "run", "build"], cwd=str(web), check=True)
    if not dist.is_file():
        raise SystemExit(f"rengu: build finished but {dist} is missing")


def _free_port(port: int, *, patterns: tuple[str, ...]) -> None:
    if not which("ss"):
        return
    result = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if f":{port} " not in line:
            continue
        m = re.search(r"pid=(\d+)", line)
        if not m:
            continue
        pid = int(m.group(1))
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.is_file():
            continue
        cmd = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if any(p in cmd for p in patterns):
            print(f"==> Stopping process on port {port} (PID {pid})...")
            subprocess.run(["kill", str(pid)], check=False)
            time.sleep(1)
            subprocess.run(["kill", "-9", str(pid)], check=False)
        else:
            raise SystemExit(f"rengu: port {port} in use by PID {pid}: {cmd.strip()}")
        break


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
    if not which("curl"):
        time.sleep(3)
        return True
    deadline = time.monotonic() + timeout
    curl_base = ["curl", "-sf", "-o", "/dev/null", "--connect-timeout", "1", "--max-time", "2"]
    if token:
        curl_base.extend(["-H", f"X-Rengu-Flow-Token: {token}"])
    attempt = 0
    while time.monotonic() < deadline:
        if subprocess.run([*curl_base, url], check=False).returncode == 0:
            return True
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
        "**/.rengu-flow-ui",
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
    cmd = args.ui_command

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
        if not args.skip_sync:
            self_heal()
            ensure_ui_dependencies()
            reexec_cli()
        _build_web(root, force=args.rebuild_web)
        host = cfg.ui.host
        port = cfg.ui.port
        _free_port(port, patterns=("rengu", "uvicorn", "rengu-flow-ui"))
        browser_host = "127.0.0.1" if host in ("0.0.0.0", "::", "*") else host
        url = f"http://{browser_host}:{port}/"
        health = f"{url.rstrip('/')}/api/v1/health"
        if not args.no_open:

            def _open_when_ready() -> None:
                if _wait_health(health, cfg.ui.token):
                    _browser_open(url)

            threading.Thread(target=_open_when_ready, daemon=True).start()
        print(f"==> Rengu UI: {url}")
        _serve(host, port, reload=False)
        return

    if cmd == "dev":
        reload_api = not args.no_reload_api
        if not args.skip_sync:
            self_heal()
            ensure_ui_dependencies()
            reexec_cli()
        _ensure_node()
        web = _web_dir(root)
        if not (web / "node_modules").is_dir():
            lock = web / "package-lock.json"
            install_cmd = ["npm", "ci"] if lock.is_file() else ["npm", "install"]
            print(f"==> {' '.join(install_cmd)} (ui/web)...", flush=True)
            subprocess.run(install_cmd, cwd=str(web), check=True)
        api_port = cfg.ui.port
        dev_port = args.dev_port
        _free_port(
            api_port,
            patterns=("rengu", "uvicorn", "rengu-flow-ui", "rengu_flow_ui", "multiprocessing"),
        )
        _free_port(dev_port, patterns=("vite", "node"))
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
        npm_env = os.environ.copy()
        npm_env["RENGU_FLOW_UI_PORT"] = str(api_port)
        npm_env["RENGU_FLOW_UI_DEV_PORT"] = str(dev_port)
        npm_cmd = ["npm", "run", "dev"]
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
        host = args.host or cfg.ui.host
        port = args.port if args.port is not None else cfg.ui.port
        _serve(host, port, reload=args.reload)
        return

    raise SystemExit(f"unknown ui command: {cmd}")

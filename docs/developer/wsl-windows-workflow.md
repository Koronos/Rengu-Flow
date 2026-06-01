# Working on Rengu-Flow from WSL (and Windows)

This repo runs on **WSL2 + NVIDIA** — training needs CUDA inside WSL, and the code lives on the
WSL ext4 filesystem. These conventions avoid the WSL/Windows pitfalls that repeatedly bite agents
and contributors.

## TL;DR
- Do **all** work inside WSL (shell, git, `uv`, training, UI). Don't drive training/scripts from
  Windows CMD/PowerShell.
- Use `uv` for everything Python.
- Line endings: the repo ships a `.gitattributes` that forces **LF** on shell scripts. On any
  Windows git that touches this repo, set `core.autocrlf=input` (never `true`).

> **For automated agents (Claude Code, etc.):** your Bash/PowerShell tools execute on the **Windows**
> side and reach the repo over the `\\wsl.localhost\ubuntu\...` UNC share. Commands that need the
> Linux venv, CUDA, or POSIX paths must be wrapped: `wsl bash -lc 'cd ~/Rengu-Flow && <cmd>'`.
> A bare `cd \\wsl.localhost\...` from the Windows shell silently lands in `C:\Windows` (Pitfall #3).

## Running things (from WSL)
| Task | Command |
|------|---------|
| Install (base) | `uv sync` |
| Install + web UI | `uv sync --extra ui` |
| CLI | `./rengu <cmd>` — or `uv run rengu <cmd>` if the launcher won't execute (see CRLF below) |
| Web UI (dev) | `uv run rengu ui dev --no-open` → Vite `http://127.0.0.1:5173`, API `:8765` |
| Tests | `uv run --extra dev pytest` (pytest is in the `dev` extra) |
| UI/Cosmos/LoKr tests | add `--extra ui --extra cosmos_predict2 --extra lycoris` |
| GPU smoke | `./scripts/run_model_smoke.sh sdxl\|sdxl_lokr\|cosmos\|cosmos_lokr` |

Open the dev UI from a **Windows browser** at `http://127.0.0.1:5173` — WSL2 forwards `localhost`,
so no extra config is needed. The FastAPI/Vite servers themselves must run in WSL (GPU + deps).

## Model paths
Copy `.env.example` → `.env` (gitignored) and set `RENGU_SDXL_CHECKPOINT_PATH` and
`RENGU_COSMOS_{TRANSFORMER,VAE,LLM}_PATH`. Training and the smoke scripts read these.

## Pitfall #1 — CRLF line endings
If `./rengu` or any `scripts/*.sh` fails with:

```
/usr/bin/env: 'bash\r': No such file or directory
```

the file was checked out with CRLF. Fixes:
- The repo's `.gitattributes` forces LF for `*.sh`, `rengu`, `*.py`, etc. After it lands, run once:
  `git add --renormalize . && git commit -m "Normalize line endings"`.
- On Windows: `git config --global core.autocrlf input` (do **not** use `true` for this repo).
- Quick local unblock without touching git: `sed -i 's/\r$//' rengu scripts/*.sh`, or just use
  `uv run rengu ...` (runs the console-script entry point, bypassing the shebang).

## Pitfall #2 — git worktrees created from Windows
A git worktree created **from Windows** gets a `.git` file pointing at a UNC gitdir:

```
gitdir: //wsl.localhost/ubuntu/home/.../.git/worktrees/<name>
```

WSL's git cannot resolve that path, so **every git command in the worktree fails** with
`fatal: not a git repository: //wsl.localhost/...`. Avoid or work around it:

- **Preferred — create worktrees from inside WSL** so the gitdir is a POSIX path and git just works:
  ```bash
  git -C ~/Rengu-Flow worktree add .claude/worktrees/<name> -b <branch>
  ```
- If you're stuck with a UNC-gitdir worktree, either do git work from the main checkout
  (`~/Rengu-Flow`), or drive git with explicit POSIX paths:
  ```bash
  git --git-dir="$HOME/Rengu-Flow/.git/worktrees/<name>" \
      --work-tree="$HOME/Rengu-Flow/.claude/worktrees/<name>" <cmd>
  ```
- Operating on the worktree with **Windows** git also trips "dubious ownership"; if you must, add
  `git config --global --add safe.directory '%(prefix)///wsl.localhost/...'`. Committing from
  Windows can also re-introduce CRLF — prefer committing from WSL.

## Pitfall #3 — running shell against UNC paths from Windows
Don't `cd` into `\\wsl.localhost\...` from Windows CMD/PowerShell — UNC paths aren't valid working
directories and the shell silently falls back to `C:\Windows`, so relative paths and scripts break.
Run the WSL shell instead.

## Pitfall #4 — testing a worktree imports the MAIN repo's code, not the worktree's
`rengu_flow` is installed **editable into the shared `.venv`, pointing at the main checkout**
(`/home/<user>/Rengu-Flow`). So `import rengu_flow` resolves to the main repo's source no matter
which worktree you run pytest from — a worktree's edits are silently invisible to its own tests.
Shadow the package with `PYTHONPATH` so the worktree's copy wins:

```bash
WT=/home/<user>/Rengu-Flow/.claude/worktrees/<name>
cd "$WT" && PYTHONPATH="$WT" /home/<user>/Rengu-Flow/.venv/bin/python -m pytest -q
```

Verify the shadow took before trusting results:

```bash
PYTHONPATH="$WT" /home/<user>/Rengu-Flow/.venv/bin/python \
  -c "import rengu_flow; print(rengu_flow.__file__)"   # must print a path under $WT
```

(The main checkout needs no shadow — `uv run pytest` there imports its own source correctly.)

## Pitfall #5 — command mangling when wrapping `wsl bash -lc` from Windows
When an agent runs `wsl -d ubuntu -- bash -lc '...'` from the Windows (Git Bash) side, the *content*
of the wrapped command gets mangled in ways that fail confusingly. All of these are avoided by
running in a real WSL shell; when you must wrap, work around them:
- **Don't rebuild `$PATH`.** `export PATH=/x:$PATH` fails with
  `export: '...Program Files...': not a valid identifier` because the inherited Windows `PATH`
  contains spaces. Call venv binaries by absolute path instead — e.g.
  `/home/<user>/Rengu-Flow/.venv/bin/deepspeed`, `.../python -m pytest`.
- **MSYS rewrites slashes.** Leading-slash args, `https://` URLs (`//whl/cu130` collapses to
  `whlcu130`), and backslashes inside the quoted `-lc` string get path-converted. For anything
  non-trivial (pip index URLs, multi-line scripts, loops), **write it to a `.sh` file and run
  `bash /home/.../script.sh`** rather than a long inline string.
- **Shell backgrounding doesn't persist.** `nohup … &` started via `wsl … bash -lc` is killed when
  the `wsl` invocation returns — use the agent tool's managed background, or an `until`-loop poller.

## Measuring VRAM under WSL2
- `nvidia-smi` reports the **whole GPU** (including the Windows desktop), and the per-process query
  (`--query-compute-apps`) returns **0** under WSL2. Judge a run's footprint by torch
  `torch.cuda.max_memory_allocated()` (the `cuda_peak_gb` in the bench log), **not** `nvidia-smi
  memory.used`.
- WSL2 **ignores** the NVIDIA Control Panel "Prefer No Sysmem Fallback" setting, so as VRAM fills the
  driver silently pages to **shared system RAM** — training keeps running but steps get erratically
  slow (e.g. a step jumping from ~13 s to ~100 s). Treat "starts spilling to shared memory" as the
  real ceiling (well below nominal VRAM), not OOM. The related allocator crash is handled by
  `configure_cuda_allocator`; low-VRAM levers and a measured 8 GB curve are in
  [vram-optimization.md](vram-optimization.md).

## Don't commit
`.env`, `data/`, `output/`, `tmp/`, and local venvs are gitignored — keep them out of commits.

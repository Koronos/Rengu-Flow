# Spec: Workflows

**Status:** Phases 0–1 (P5-1, P5-2) **implemented** 2026-08-08/09; Phase 2 (P5-3, retiring
`kind='prep'`) proposed. **Date:** 2026-08-09. **Backlog:** [P5-3](../BACKLOG.md).

> **Shipped behaviour is documented elsewhere.** This page is the design and the rejected
> alternatives — *why* each rule exists. For what the feature does today, read
> **[user/workflows.md](../user/workflows.md)**; for the technical contract and where the code
> lives, **[developer/workflows.md](../developer/workflows.md)**. Where this spec and those two
> disagree, the code is the source of truth and those two follow it.

**Scope:** control plane (`rengu_flow_ui/` + `ui/web/`). The training core is untouched; the prep
engine gained exactly one rule — `validate_for_stage` now refuses a `tag` stage with no models,
which previously loaded nothing, wrote nothing and exited 0.

Rengu Flow has two disconnected ways to act on a dataset folder. **Toolbox**
([user/toolbox.md](../user/toolbox.md)) runs user-authored Python functions but has no typed
output — the only channel is stdout, so tools cannot be chained. **Prep**
([user/dataset-prep.md](../user/dataset-prep.md)) has five stages (`tag`, `caption`, `clean`,
`quality`, `index`) launched as one-off `kind='prep'` rows on the *same single-runner queue as
training runs*; chaining tag → caption today is a hardcoded `chainCaption` toggle inside a
1640-line form.

This spec defines **Workflows**: a composable chain of nodes — user-authored (Toolbox) and
predefined (prep stages, training launch) — where each node's output folder is the next node's
input, with one saved state per workflow, resumable from any step. See
[developer/web-ui.md](../developer/web-ui.md) for the control-plane layout this extends.

---

## Goal

Let a user build, save, clone and re-run a dataset pipeline without re-entering configuration.
The guiding use case, stated by the user:

> *"If I need to re-tag another folder with the same configuration, I should only have to change
> the input folder."*

Three concrete outcomes:

1. **Composition.** `folder → clean → quality → tag → caption → train` is one saved object, not
   six manual form submissions.
2. **Resumability.** Each node stores its last output. Starting from step 4 reuses steps 1–3's
   saved outputs; no re-run, no re-typing.
3. **Lane separation.** Prep leaves the training queue. A CPU-only node (`quality` with
   `metric = "blur"`) runs *while* a training job holds the GPU, arbitrated by an explicit lease
   instead of by queue exclusion.

## Non-goals

- **No run history.** One state per workflow (explicit user decision). Re-running overwrites.
  This is the deliberate difference from `runs`/`jobs`, which do keep history.
- **No draggable canvas.** No graph library, no new frontend dependency. Nodes are an
  auto-laid-out vertical chain, configured in a side panel — the Azure Logic Apps / Step Functions
  model, not the ComfyUI model.
- **No parallel DAG execution.** One node at a time, one active workflow at a time.
- **No branching, conditionals, or loops.** Not requested; not speculatively built.
- **No per-device queue slots.** `job_queue.has_active_runner()` stays global — see
  [Risks and known limits](#risks-and-known-limits).
- **No type system on connections.** One payload type, always compatible.

---

## Concepts

**`DatasetHandle`** — the only value that travels between nodes:

```ts
{ path: string, caption_format: "sidecar" | "json", caption_ext: string }
```

Every node consumes a handle and emits a handle. Connecting is always valid.

**Node** — one executable step. Declares `from` (which node's handle it reads), its own config,
and its GPU policy.

**`from`** — a reference to another node's handle. **Invariant: `from` may only reference a node
that appears earlier in the list.** This single rule makes cycles structurally impossible (no
cycle detection, no topological sort) and means **execution order is list order**. `from` decides
*which folder a node reads*, never *when it runs*. Non-consecutive links (node ① feeding node ③)
are just a `from` that skips.

**Variables** — workflow-level string constants referenced from node config. Configuration only;
they never capture node output (that is what `from` is for).

**State** — one saved record per workflow: per-node status, saved output handle, exit code,
timestamps, and the config hash that was in effect when the output was produced.

---

## Node catalog

Eight types. The output rule per type is the part most likely to be got wrong, so it is normative
here.

| Type | Consumes | Emits | GPU by default |
|---|---|---|---|
| `folder` | — | its literal config as a handle | no |
| `prep.tag` | handle | **the same handle** — writes tag sidecars in place | yes |
| `prep.caption` | handle | **the same handle** — writes caption lines in place | yes |
| `prep.clean` | handle | `in_place ? input : (output_dir or <input>/cleaned)` | yes |
| `prep.quality` | handle | **the same handle** — the survivors | `metric != "blur"` |
| `prep.index` | handle | **the same handle** — the SQLite index lives outside the dataset | yes |
| `tool` | handle (optional) | derived from the function's **return value** | no |
| `train` | handle | nothing (terminal, fire-and-forget) | takes no lease itself |

### The `output_dir` trap — normative

`clean` and `quality` both write `report["output_dir"]`, and **they mean opposite things**:

- `cleanup.clean_folder` sets it to the *result* folder — `config.output_dir` or `<src>/cleaned`
  (`rengu_flow/prep/cleanup.py:270-272`).
- `filter_folder` sets it to the *quarantine* folder — `config.output_dir` or
  `<src>/low_quality` (`rengu_flow/prep/quality.py:179`, `:191`). The resulting dataset is still
  the **input** folder.

A generic "read `report['output_dir']` as the output" would make a `quality → caption` workflow
caption the reject pile. **`collect_output` consults `output_dir` for `clean` only.** No change
to `rengu_flow/prep/` is required; the UI additionally relabels the `quality` field to
**"Quarantine folder"**.

### Per-type notes

- **`folder`** — the only node without a `from`. It is where `${dataset_dir}` lives, which is
  what makes the guiding use case a one-field edit. Marked `done` by validating the path exists;
  no subprocess.
- **`prep.clean` with `copy_undetected = false`** emits a folder containing **only** cleaned
  images. The form must warn: *"The emitted folder will contain only cleaned images; images with
  nothing detected are left behind."*
- **`prep.quality`** GPU default follows `metric`: `blur` is a pure-CPU Laplacian variance with no
  extra dependencies; `aesthetic` and `iqa` load models. The user can override.
- **`prep.index`** has **no form today** — `stageLabel` does not map it and `buildConfig()` has no
  branch in `PrepJobFormView.vue`. `IndexStageForm.vue` is new construction, not extraction.
- **`train`** is terminal and fire-and-forget: it marks `done` when the job is *enqueued*, not
  when training finishes. See [Risks and known limits](#risks-and-known-limits).

**Not built:** `sink` (the result is the last node's handle, shown in the run bar), `note` /
`passthrough` (covered by per-node `description`), `branch` / `condition` / `loop` (not
requested). A "save to dataset library" node is defensible but the `train` node already covers
the reason it would exist — deferred to [Open work](#open-work).

---

## Graph model

The graph is JSON, not TOML: it is a heterogeneous nested list, and unlike the rest of the
library it never feeds the trainer.

```jsonc
{
  "version": 1,
  "name": "Re-tag character set",
  "variables": [
    { "name": "dataset_dir", "value": "D:/datasets/aoi", "description": "Folder to process" }
  ],
  "nodes": [
    {
      "id": "n1", "type": "folder", "title": "Source folder",
      "from": null, "enabled": true,
      "config": { "path": "${dataset_dir}", "caption_format": "sidecar", "caption_ext": ".txt" },
      "gpu": { "required": false, "wait": true, "device": null }
    },
    {
      "id": "n2", "type": "prep.tag", "title": "Tag",
      "from": "n1", "enabled": true,
      "config": { "models": ["pixai-v0.9", "cl-tagger-1.02"], "overrides": {}, "max_tags": 255 },
      "gpu": { "required": true, "wait": true, "device": 0 }
    }
  ]
}
```

- **`id`** is opaque and stable, minted client-side and never reused after a delete. It is *not*
  the index — reordering would silently re-point every `from`.
- **`config`** holds exactly the stage section (`TagStageConfig`, `CaptionStageConfig`, …) from
  `rengu_flow/prep/config.py` **minus `path` / `caption_format` / `caption_ext`**. Those three are
  injected by the executor from the incoming handle. That omission is what makes "change the input
  folder" a single edit in a single node.
- **Parsing is tolerant**, mirroring `parse_prep_config`'s `_fill_dataclass`: unknown keys are
  logged and ignored, never fatal. An unknown *node type* is fatal at execution but **preserved on
  save**, so downgrading the app never destroys the user's graph.
- **Validation** rejects a graph where any `from` points forward or at itself, **and where `from`
  is `null` on any type other than `folder` or `tool`**. The forward rule alone is not enough:
  deleting a `folder` node splices its children to *its* `from`, which is `null`, so a `prep.clean`
  node ends up sourceless, passes pre-flight, and dies mid-run on
  `validate_for_stage`'s `"Prep config needs a dataset 'path'"` — after the earlier nodes already
  ran, in direct contradiction of the promise that pre-flight reports every error up front.
  Deleting a `folder` with children is a distinct prompt: *"③ has no source. Pick a new source
  folder first."*
- **`id` is minted with `crypto.randomUUID()`** (or an 8-char random suffix), never `n<max+1>`.
  With two tabs open, sequential minting hands the same `n5` to two different node types; the
  second save wins and inherits a `state_json["n5"]` written by a node of another stage, so
  `collect_output` reads someone else's `report.json`.
- **`PUT /workflows/{id}` is optimistically concurrent** (a `version` column, 409 on mismatch) and
  **returns 409 outright while `state_json.status` is `running` or `cancelling`**; the editor goes
  read-only with a "Stop to edit" banner. Without the running guard, deleting the node that is
  currently executing removes it from `content`, so the runner never finalizes it, never releases
  its lease, and immediately launches the next node — two nodes running at once, a workflow stuck
  in `running` forever, and a Stop button with nothing to stop. Reordering is equally unsafe: list
  order is execution order but is not part of any hash, so a `Move up` on a `done` node changes the
  pipeline while every card still reads fresh. Workflows are the one object in the app with **no
  history** (see Non-goals), so a lost-update here is unrecoverable.
- **Variable resolution is a single pass, no recursion** — a variable whose value contains
  `${other}` is left as-is.
- **Pre-flight also asks "would this actually launch?"** For every enabled `prep.*` node,
  `validate()` materializes the resolved config and runs the same
  `PrepConfig.validate_for_stage(stage)` the launcher runs. Structure alone is not enough: a
  `prep.index` node with an empty config — literally what "Add step" produces — passed a
  structural check with zero errors and then died in `_build_prep_launch` *after* the earlier
  nodes had already done their work, which is exactly the mid-run surprise this section promises
  not to have. Two wrinkles: `validate_for_stage` also demands an existing `path`, which in a
  workflow arrives from the edge and cannot be known until the upstream node has run, so a
  placeholder directory is injected and path rules are left to the launch; and a node that already
  reported an unknown variable is skipped, because the materialized config is not the one that
  would run.
- **A node is born with this app's form defaults, never an empty config.** With `config: {}` the
  server materializes the *dataclass* defaults, which are not what the UI shows — a `prep.tag`
  node would run two taggers at `max_tags: 255` while its card read "no tagger selected" and its
  form showed 40. `createNode` writes the form's own defaults in at creation so that what the user
  sees is what runs, whether or not they ever opened the step.

### Execution order

List order. Strictly sequential, one node at a time, even where two chains are independent —
consistent with the single-runner model. Disabled nodes are skipped; a node whose `from` is
disabled reads that node's *saved* handle if present, otherwise fails validation.

---

## Variables

- **Syntax** `${name}`, names matching `[A-Za-z_][A-Za-z0-9_]*`. **Escape** `$$` → literal `$`.
  One rule, no backslashes.
- **Where they apply:** string fields only — paths, model ids, prompts, output dirs, and
  `text`/`textarea`/`select` tool inputs. Never numbers or booleans. Substituting text into a
  numeric field forces coercion rules and a new class of type error; "variables are text in text
  fields" is a rule nobody has to learn.
- **When they resolve:** at execution, per node, immediately before launch. The saved graph keeps
  `${dataset_dir}` literal.
- **Missing variable:** the workflow **does not start**. Pre-flight returns every error at once
  (`node n3 · quality.output_dir → unknown variable ${outdir}`) and the Run button is disabled
  with the count. Never substituted with an empty string.
- **Source of truth** is the server (`workflow_graph.resolve_variables`). The client carries the
  same ~25-line function in `src/lib/workflowVars.ts` purely for the inline preview, tested on
  both sides; a round-trip per keystroke costs more than the duplication.

---

## Execution model

### Two lanes

| Lane | Runner | Holds |
|---|---|---|
| Training | `job_queue.try_start_next()` — one job at a time, unchanged | a GPU lease while running |
| Workflow | `workflow_runner.tick()` — one workflow, one node at a time | a GPU lease per GPU node |

The lanes are independent; the GPU lease is what keeps them from colliding. A CPU-only node never
takes a lease and therefore runs alongside training.

### The tick lives in `queue_poller`, not a new thread

`rengu_flow_ui/queue_poller.py:32` is already a daemon thread with clean shutdown
(`Event.wait`) and a `try/except` that keeps the loop alive across a bad pass. `_tick()` gains a
second call:

```python
def _step(step, label) -> None:
    """One guarded step. BaseException, not Exception: ensure_profiles raises SystemExit,
    and this thread owns reap_dead() plus the whole workflow lane."""
    try:
        step()
    except KeyboardInterrupt:
        raise
    except BaseException:
        _logger.exception("%s failed", label)

def _tick() -> None:
    _step(gpu_lease.reap_dead, "gpu lease reap")     # before either lane can acquire
    _step(jobs.refresh_all_jobs, "queue poller tick")
    _step(workflow_runner.tick, "workflow tick")     # Phase 1
```

One `try` per step, not one around all three: a `reap_dead` that raises must not take down the
run reconciliation that works today without any of this code.

Two reasons, both concrete:

- **One writer thread on `jobs.db`.** A second thread ticking every few seconds doubles
  contention exactly while a training run is writing — the problem the pragma comments in
  `db._apply_connection_pragmas` already document.
- **`gpu_lease.reap_dead()` must run once per tick, before either lane tries to acquire.** A
  single ordered tick guarantees that without any cross-thread coordination.

`tick()` is also called synchronously at the end of `POST /workflows/{id}/start` and `/cancel`, so
the UI sees the effect without waiting for the interval — the same pattern as
`job_queue.start_job_immediately`.

**`tick()` therefore needs a non-blocking lock.** FastAPI runs sync endpoints in a threadpool, so
two `/start` calls — a double click, or two tabs — execute it concurrently, and the poller is a
third thread. Two threads would both see the same node `pending` and both `popen` it; for a
CPU-only node there is no lease to serialize them, so two `prep.quality` processes would move
files into `low_quality/` at the same time and the second `pid` write would orphan the first
process forever. `tick()` opens with `if not _tick_lock.acquire(blocking=False): return`,
mirroring `job_queue._start_lock`.

Likewise **all `state_json` mutation goes through one `workflow_db.mutate_state(wf_id, fn)`** that
does the read-modify-write inside a single transaction with a compare-and-swap on the previous
value. Without it, a `/cancel` handler setting `cancelling` and a poller writing progress race on
a JSON blob, last write wins, and the Stop the user just clicked disappears.

### A node is not a job

Nodes do **not** go in the `jobs` table. Adding a third `kind` value while removing the second is
self-defeating, and a node has no `config_path` of training shape, no `run_dir` of output shape,
and no `num_gpus`. The runner spawns the subprocess directly with
`subprocess_util.popen_repo_subprocess`, which already accepts `env=` and `log_header=`.

### Node lifecycle

```
pending ──▶ waiting_gpu ──▶ launching ──▶ running ──▶ done
   │             │                           │    └──▶ failed
   │             │                           └──▶ stopping ──▶ stopped
   └──────── skipped (disabled)
```

**`launching` is written to `state_json` before `popen`, not after.** Without it, a server killed
in that window leaves the node `pending` while its process — detached by design — keeps running;
on restart the runner launches a *second* `prep.tag` and two taggers write line 1 of the same
sidecars concurrently, interleaving captions across the dataset with neither process aware. A
`launching` node is never auto-started on reconcile: it is failed with *"Interrupted while
starting; check for an orphan process"*, or adopted if its `node.log` is growing.

**Prep extras are installed before the spawn, not by it.** `rengu prep <stage>` calls
`ensure_profiles(["prep"])` → `uv sync --inexact --extra prep`
(`rengu_flow/cli/prep_cmd.py:228-230`, `rengu_flow/install/profiles.py:120-129`). Today that can
never overlap a training run, because the shared queue is what prevents it — the very guarantee
this feature removes. A first-ever `prep.quality` node advertised as "CPU-only, runs alongside
training" would rewrite `site-packages` under a live DeepSpeed process and hand it an `ImportError`
hours in, at checkpoint time; on POSIX it also reinstalls the editable project the trainer is
running from.

So `_launch_node` runs `ensure_profiles(["prep"])` **in the UI process, with the node in
`launching`** — and refuses **only when the profiles are actually missing and a run is active**.
The condition matters: the hazard is the `uv sync` write, not the node. Refusing on
`has_active_runner()` alone would mean no prep node ever runs during a training run, which is the
exact opposite of Goal 3 — a CPU-only `quality` node is supposed to run *while* training holds the
GPU. Once the extras are installed there is no write, no hazard, and no reason to wait.

A node refused this way returns to `waiting_gpu` with a readable reason **and releases its lease
while it waits** — holding a GPU token to wait on a dependency install would deadlock the very
lane it is queuing behind.

**"A run is active" must be read from the lease, not from `has_active_runner()`.** This looks like
a detail and is not: `try_start_next` acquires the lease and *then* calls `jobs.start_job` →
`ensure_training_extras` → its own `uv sync`, so for the minutes that install takes the job row is
still `pending` and `has_active_runner()` returns `False`. Checking the rows would therefore green-
light a `uv sync --extra prep` at exactly the moment a training run is inside its own — the precise
collision the guard exists to prevent, reproduced in review. Query `gpu_lease.snapshot()` for a
`holder_kind == "train"` holder instead: the lease is taken *before* `ensure_training_extras`, so
it covers the blind window. Keep the row check too — a run whose lease was reaped is still running,
and then the rows are the only witness.

The claim that device selection needs no change to the prep engine still holds; the claim that
*nothing* about prep changes does not.

`_launch_node` validates *before* spawning — `PrepConfig.validate_for_stage(stage)` turns "folder
does not exist" into a clean `error` on the node row instead of a traceback buried in `node.log`.
This is the same check `prep_routes.create_prep_job` performs today.

### Node directory

`data/workflows/<workflow_id>/<node_id>/`, shaped exactly like today's `data/prep/<job_id>/` so
signal files, `report.json`, `@@RFPROG@@` markers and log tailing work with no engine change:

```
node.log      stdout+stderr, with the "--- rengu-flow-ui ---" header
prep.toml     prep nodes: the materialized TOML with the injected path
report.json   prep nodes: written by run_stage on every exit path
tool.py       tool nodes: copy of the user script
inputs.json   tool nodes: cast kwargs
_runner.py    tool nodes: generated PEP 723 shim
result.json   tool nodes: the return value (new — see Toolbox contract)
quit          signal file, present only while cancelling
```

The path is a function — `workflow_db.node_dir(wf_id, node_id)` — not a literal in five places, so
adding on-disk history later (`<node_id>.<timestamp>`) stays a one-line change.

### Restart reconciliation

Node subprocesses **survive** a server restart: `popen_kwargs_new_group()` puts them in their own
process group and stdout goes to a file, not a pipe. This is why a prep job survives a restart
today. Toolbox does the opposite — it loses the in-memory `_active` handle and marks a
still-running process `failed` (`rengu_flow_ui/toolbox.py`, `run_status`). **That behaviour is not
replicated.**

Reconciliation matches on `pid` **and `pid_create_time`**, not `pid` alone:

```python
def _still_alive(node) -> bool:
    if node.pid is None or not pid_alive(node.pid):
        return False
    if node.pid_create_time is None:
        return True
    return abs(psutil.Process(node.pid).create_time() - node.pid_create_time) < 1.0
```

A workflow can sit in `running` for weeks (the user shuts the machine down mid-prep). After a
reboot, PID 4231 is some unrelated process; without the create-time check the node stays
"running" forever **and its GPU lease is never released**, permanently blocking the training lane.
Three lines remove the only deadlock in the system.

On start: `reap_dead()`, finalize any `running`/`stopping` node whose process is gone, and reset
`waiting_gpu` nodes to `pending` so the lease is re-evaluated from scratch.

### Cancellation — graceful, then hard

Because a tick comes back to look again, cancellation can be two-phase for real:

1. `POST /workflows/{id}/cancel` sets `cancelling` and calls `tick()`.
2. First pass: drop `SIGNAL_QUIT` in the node dir and set the node `stopping`. Prep stages check
   it between batches/chunks via `runner._make_should_stop`, so a half-done `tag` finishes its
   current batch and writes `report.json` instead of losing the work.
3. After a 20 s grace period: `terminate_process_tree(pid)`.

This differs deliberately from `jobs.stop_job`, which signals and kills in the same call because
nothing revisits it.

- **Tool nodes** have no signal contract (by decision: no complexity added to the tool model) →
  terminated directly, no grace. Documented as: *a cancelled tool is killed; if it needs cleanup,
  use `try/finally`.*
- **`train` nodes** have nothing to stop — the enqueued job is fire-and-forget and is stopped from
  the training queue.

Stale signal files are swept when a node starts, exactly as `prep_jobs.requeue_prep_job` does
today. Without that, re-running a cancelled node exits immediately.

---

## GPU arbitration

### The lease table

```sql
CREATE TABLE IF NOT EXISTS gpu_leases (
    device          INTEGER PRIMARY KEY,   -- physical index; -1 = host with no enumerable GPU
    holder_kind     TEXT NOT NULL,         -- 'train' | 'workflow'
    holder_id       TEXT NOT NULL,         -- 'job:42' | 'wf:7:n2'
    pid             INTEGER,
    pid_create_time REAL,
    acquired_at     TEXT NOT NULL
);
```

`PRIMARY KEY (device)` **is** the mutex: inserting over an occupied device raises
`sqlite3.IntegrityError`. That is a free, atomic compare-and-swap, correct even across processes.

In-process memory was rejected: a prep node survives a server restart *still holding the GPU*, and
memory cannot describe a resource that outlives the process containing it. A lock file was
rejected because it means re-inventing `O_EXCL`, parsing and stale cleanup that SQLite already
provides.

### API

```python
def enumerate_devices() -> list[int]                       # cached per process
def reset_device_cache() -> None                           # tests + driver reload
def acquire(holder_kind, holder_id, devices) -> bool
def bind_pid(holder_id, pid, pid_create_time=None) -> bool | None  # tri-state, see below
def release(holder_id) -> None
def reap_dead() -> list[str]
def reconcile_on_start() -> list[str]
def snapshot() -> list[dict]
def wait_reason(devices) -> str
```

`devices = None` means *auto / host-exclusive* and inserts **every** enumerated device in one
transaction (`with conn:` — an autocommit `executemany` would leave a half-acquisition on
conflict), so "auto" and "GPU 1" share one code path: a multi-row insert that either lands whole
or fails whole. On a host with no `nvidia-smi`, a single `device = -1` row is inserted and
arbitration still works.

**`acquire` validates every requested device against `enumerate_devices()`** and falls back to
host-exclusive when an index is not enumerated. Without this, a host where enumeration failed
caches `[-1]`, an "auto" holder takes `-1`, a node pinned to `device: 0` takes `0`, the two do not
conflict, and both land on the same physical GPU.

### Reaping — by holder validity, not by timing

**This is the load-bearing rule.** A lease is not freed because a timer expired; it is freed
because its holder is provably gone.

```
reap_dead()          — free 'job:<id>' when the row is missing or its state is terminal
                     — free any lease whose bound pid is dead, a zombie, or reused
reconcile_on_start() — reap_dead(), plus free every lease with pid IS NULL
```

The reasons each clause exists:

- **Terminal-holder reaping is what makes the training lane leak-proof.** A job reaches a terminal
  state through at least five paths that never touch `jobs.poll_job`: `stop_job` with `pid is
  None` (`jobs.py:108-110`, reachable because `POST /jobs/{id}/stop` does not guard on state),
  the config-missing branch in `try_start_next` (`job_queue.py:195-197`), `delete_job_record` and
  `dequeue_job` on a `pending` row (`job_queue.py:663-688`), and a hard server kill. Sprinkling
  `release()` at five call sites is a checklist that will be wrong the first time a sixth path is
  added; reconciling against the job row is correct by construction. `poll_job` still releases
  eagerly — that is a latency optimization, not the correctness mechanism.
- **No `pid IS NULL` timer.** The window between `acquire` and `bind_pid` contains
  `ensure_training_extras` → `ensure_profiles` → `uv sync --inexact --extra …`
  (`rengu_flow/install/manager.py:88-96`), which on a cold extra takes **minutes, not
  milliseconds**. A 60-second timer would free the lease mid-install, let a workflow GPU node
  start, and then let training spawn with no lease at all — reintroducing the exact regression
  this table exists to prevent. A `pid IS NULL` lease is legitimate for as long as its job row
  says `pending`.
- **`reconcile_on_start` frees `pid IS NULL` leases** because an unbound lease cannot survive the
  process that created it: the launch it belonged to did not happen.
- **Zombies count as dead.** `platform_compat.pid_alive` is `os.kill(pid, 0)` on POSIX, which
  returns `True` for a zombie, and nothing ever `wait()`s these detached children. Without an
  explicit `psutil.STATUS_ZOMBIE` check a finished node holds its lease indefinitely.
- **`bind_pid` records `pid_create_time`** (captured internally via `psutil.Process(pid)`), and
  `reap_dead` applies the same `abs(create_time - stored) < 1.0` test as node reconciliation.
  Without it the reboot deadlock the node design spends fifteen lines eliminating simply moves
  into the lease table. `bind_pid` returns `False` when it updates zero rows — the lease was
  reaped underneath it — and the caller must then kill what it just spawned.

No timeouts are applied to live processes: an 8-hour caption run is legitimate.

### Both lanes participate

**Training** — in `job_queue.try_start_next()`, inside the existing `_start_lock`, the acquire
goes **after** the `config_path` validation and immediately before `jobs.start_job`. Placing it at
the top of the block leaks the lease on the config-missing early return:

```python
job = pending[0]
if not job.config_path or not Path(job.config_path).is_file():
    db.update_job(job.id, state="failed", finished_at=_now(), exit_code=-1)
    return db.get_job(job.id)          # acquire above this line leaks permanently
holder = f"job:{job.id}"
if not gpu_lease.acquire("train", holder, _devices_for_job(job)):
    return None                        # stays pending; the tick retries
try:
    pid = jobs.start_job(job)
except BaseException:                  # NOT Exception: ensure_profiles raises SystemExit
    gpu_lease.release(holder)
    raise
if gpu_lease.bind_pid(holder, pid) is False:   # tri-state; None means "no pid to bind"
    terminate_process_tree(pid)                # the lease was reaped underneath us
    db.update_job(job.id, state="failed", exit_code=-1, pid=None)
    return db.get_job(job.id)
return db.get_job(job.id)
```

```python
def _devices_for_job(job) -> list[int] | None:
    # num_gpus >= 2 is always host-exclusive: DeepSpeed enumerates everything.
    if job.num_gpus == 1 and job.gpu_index is not None:
        return [int(job.gpu_index)]
    return None
```

`jobs.poll_job()` releases right before the terminal `update_job` — its single terminal exit.

**Workflow** — acquire in `_launch_node`, release in `_finalize_node` under `finally`.

**The tick gains an unconditional `try_start_next()` — in Phase 1, not Phase 0.** Today nothing
calls it periodically: `refresh_all_jobs` iterates only `running`/`stopping` jobs, so
`try_start_next` fires solely on a terminal transition (`jobs.py:159-162`), from the explicit Start
endpoint, or from `start_job_immediately`. Once the workflow lane can hold the GPU while *no job
is running*, a failed acquire leaves the job `pending` with **no retry path at all** — the queue is
silently dead until the user clicks Start.

That retry path is **not** needed in Phase 0, and adding it there is a behaviour change disguised
as groundwork. In Phase 0 the only possible holder is the running training job, and then
`has_active_runner()` is already `True`; the state "pending, could start, but the acquire failed"
does not exist. Meanwhile the unconditional call breaks a guarantee documented in two places —
*"a bare refresh that finds nothing active **never starts an idle queue** — the first run is always
started explicitly by the user"* (`jobs.py:254-256`) and *"Add to the pending queue only — do NOT
start"* (`job_queue.py:495-496`). A user who enqueues a run and walks away would come back to
DeepSpeed already running, within one poll interval.

So Phase 1 adds the call **and** updates both contract comments and the user docs, with a test
fixing the new semantics. Phase 0 leaves the poller's start behaviour exactly as it is.

Two constraints on the Phase 1 version, because a tick that runs every three seconds must not
cost anything or overlap anything:

- **It must not add load that could degrade a training run.** `try_start_next` opens with
  `has_active_runner()` → `db.list_jobs(limit=500)`, a full scan every tick. Guard it with a cheap
  indexed `SELECT 1 FROM jobs WHERE state='pending' LIMIT 1` and return immediately when the
  queue is empty — which is the common case, so the usual tick cost stays one trivial query.
- **It cannot overlap.** `try_start_next` is already serialized by `_start_lock` and gated by
  `has_active_runner()` **and** the GPU lease, so a periodic call adds a retry, never a second
  runner. The regression test is that two ticks in a row with one pending job start exactly one
  process.

`has_active_runner()` is unchanged — "one training run at a time" is a separate rule from GPU
ownership and stays true even on a multi-GPU host.

### The poller must survive `SystemExit`

`ensure_profiles` raises `SystemExit` when uv is missing or a profile stays unimportable
(`rengu_flow/install/manager.py:105-110`). `queue_poller._tick` catches `Exception`, and
`SystemExit` derives from `BaseException` — so one bad job kills the thread silently. Today that
costs "the queue stops draining until restart". Once this thread also owns `reap_dead()` and the
workflow lane, the same bad job means no lease is ever freed and **neither lane can ever acquire
again**. `_tick` catches `BaseException` (re-raising `KeyboardInterrupt`).

### Per-node policy

```jsonc
"gpu": { "required": true, "wait": true, "device": 0 }
```

- `required` — defaults per type (see the catalog); for `prep.quality` it follows `metric`.
- `wait` — `true` (default) joins the GPU wait queue. `false` starts immediately without asking
  for a lease: the explicit escape hatch for running several things on one GPU when the user knows
  they fit.
- `device` — `null` (auto) or a physical index.

A waiting node shows `waiting_gpu` with a readable reason: *"Waiting for GPU 0 — held by training
job 42."*

### Device selection (new capability)

There is no GPU selection in Rengu Flow today — only `num_gpus` (a count) and a **commented-out**
`CUDA_VISIBLE_DEVICES = "0"` in `rengu.local.toml.example:41`.

- **Enumeration** reuses `rengu_track/system_stats.py`, which already runs
  `nvidia-smi --query-gpu=index,name,...`. A thin `list_gpu_devices()` is exposed;
  `gpu_lease` caches it **per process** — topology does not change while the server lives, and the
  lease sits in the tick's hot path where spawning `nvidia-smi` every second is unacceptable. The
  UI picker reads the system-stats WebSocket that `HostStatsBar.vue` already consumes: **no new
  endpoint**.
- **Application** is `CUDA_VISIBLE_DEVICES` in the child env, merged **last** so the node's choice
  beats `[training.env]` from `rengu.local.toml`. `jobs.start_job` already accepts an `env=`
  parameter that no caller passes today (`rengu_flow_ui/jobs.py:59-63`, merged at `:88-89`) — the
  injection point exists with zero refactor.
- With `CUDA_VISIBLE_DEVICES=3` the child sees that GPU as `cuda:0`, and all of `rengu_flow/prep/`
  uses the default device. **Device selection therefore requires no change to the prep engine**,
  and the nested `uv run --with` overlays used by `quality`/`index` inherit the variable for free.
- Prep nodes must **not** receive `training_subprocess_env()` — NCCL, TF32 and allocator knobs are
  training concerns they only inherit today because `start_job` is shared. A node's environment is
  the inherited `os.environ` + `PYTHONUNBUFFERED=1` + `CUDA_VISIBLE_DEVICES` when applicable.
- **Training** gains a `gpu_index` column on `jobs`. With `num_gpus >= 2` nothing is set
  (DeepSpeed enumerates everything), preserving today's behaviour exactly. Adding it is **not
  just a line in `_JOBS_ADDITIVE_COLUMNS`**: it also needs a field on the `JobRecord` dataclass
  (`db.py:52-80`), a read in `_row_to_job` (`:211-238`), a column in `create_job`'s `INSERT`
  (`:314-339`), and an entry in `update_job`'s `allowed` set (`:370-388`) — which today does not
  even contain `kind`.

---

## Persistence

One additive table, registered from `library_db.init_library_tables` the same way `datasets` is:

```sql
CREATE TABLE IF NOT EXISTS workflows (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL DEFAULT '',   -- denormalized copy of the graph's own `name`
    content    TEXT NOT NULL,               -- graph JSON: nodes, variables, from
    state_json TEXT NOT NULL DEFAULT '{}',  -- the single state
    version    INTEGER NOT NULL DEFAULT 0,  -- optimistic concurrency on content
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

**Two columns, not one blob, and not normalized tables.** The editor writes `content`; the
executor writes `state_json`. They never collide, so saving an edit cannot clobber live progress
and progress cannot clobber an edit — which is precisely what makes "edit node 3, save, and nodes
1–2 stay `done` with their outputs" work.

A normalized `workflow_nodes` + `workflow_edges` schema was considered and rejected: with
sequential execution and one active workflow, the tick reads one small row every 1–3 s.
Normalizing buys a hot path that does not exist and costs graph↔row synchronization on every save.

**The name lives in the graph; the column is a copy `PUT` keeps in step.** There is no rename
route — the UI renames by saving a graph whose `name` changed — so `PUT /workflows/{id}` passes the
parsed graph's name to `update_graph`, which writes it in the same compare-and-swap as `content`.
The column exists only so the list view can sort and search N workflows without parsing N blobs;
left to drift, a rename saved in the editor never reached the list and there was nothing that could
ever correct it. `workflow_db` still treats `content` as an opaque string: it is handed the name,
not made to parse it. `POST /clone` rewrites the graph's `name` to the copy's before the row lands,
so the copy is not renamed back to its source's name by its own first save.

### `state_json`

```jsonc
{
  "status": "idle | running | cancelling | failed | stopped | done",
  "current_node": "n2",
  "started_at": null, "finished_at": null,
  "nodes": {
    "n2": {
      "status": "pending|waiting_gpu|launching|running|stopping|done|failed|stopped|skipped",
      "pid": 4231, "pid_create_time": 1754689201.4, "exit_code": 0,
      "started_at": "...", "finished_at": "...",
      "output": { "path": "...", "caption_format": "sidecar", "caption_ext": ".txt" },
      "saved_input": { "path": "...", "caption_format": "sidecar", "caption_ext": ".txt" },
      "config_hash": "…", "error": "",
      "adopted": false, "log_size": 0, "stop_requested_at": null, "result": null
    }
  },
  "queue_claim": { "job_id": 42, "node_id": "n5" }
}
```

`saved_input` is not decoration: the staleness rule compares it, and a node finalized without it
makes every input comparison read `None != handle` and marks the whole chain stale. `queue_claim`
is what stops two workflows from reordering each other's front-of-queue run.

The output handle is stored **resolved**, not as a reference to the upstream node. That is what
makes starting from an intermediate step possible without re-running anything.

### Schema version

**`SCHEMA_VERSION` is not bumped.** Under the policy in `rengu_flow_ui/db.py:30-38`, a bump is for
*incompatible* changes — a column removed, renamed, or with changed semantics — and it discards
the user's local libraries. Everything here is additive:

- New tables via `CREATE TABLE IF NOT EXISTS`, healed in an existing DB exactly as
  `init_library_tables` was.
- `gpu_index` on `jobs` → a new column with a `NULL` default, added to `_JOBS_ADDITIVE_COLUMNS`
  and healed by `_reconcile_jobs_columns` (`db.py:121-143`).
- Retiring `kind='prep'` removes **neither the column, nor its default, nor its semantics** — the
  code simply stops *writing* that value. Existing rows stay readable and correct.

### Clone

Clone copies `content` and **discards `state_json`**. A clone never inherits another workflow's
outputs.

---

## Staleness

Each node has a `config_hash` over `(HASH_VERSION, type, the **materialized** stage config with
variables resolved and defaults filled from the current app version, gpu, config_hash of its
`from` node)`.

Because the parent's hash feeds the child's, **downstream invalidation is free — there is no
propagation code**. The saved state records the hash that was in effect when the output was
produced, plus the input handle it actually consumed:

```
stale(n) = n has a saved output
           && (current_hash(n) != saved_hash(n) || effective_input(n) != saved_input(n))
```

**The hash is computed on the server only.** The client renders `stale` from the state payload and
never recomputes it. This is not a style preference: `QualityStageConfig.blur_threshold` defaults
to `80.0`, and `JSON.stringify` emits `80` where `json.dumps` emits `80.0` — every workflow with a
`prep.quality` node would disagree across the wire. Integral floats are only the first divergence
(`1e21`, `-0`, `NaN` follow).

**Hashing the materialized config, not the stored partial dict**, is what makes the rule survive a
release: a form that gains a field would otherwise turn every saved node in every workflow amber
on upgrade day. `HASH_VERSION` covers the inverse case — a *changed default* silently altering
behaviour while the hash holds still — and a per-workflow **"Accept current configuration"** action
rewrites `saved_hash = current_hash` on every `done` node. Without that escape hatch one release
costs the user a full re-run or a wall of amber rings they learn to ignore, which destroys the
only staleness signal there is.

**Folding the saved input handle in** covers what the config chain cannot see: a tool that returns
a computed folder (`f"{path}/export-{ts}"`) leaves its downstream node's config textually
unchanged, so a config-only rule would mark it fresh, `Run` would skip it, and the run bar would
report a new empty folder as the result.

Changing a variable enters through the same door: the hash is computed over the *resolved* config,
so editing `dataset_dir` marks the `folder` node stale and, by inheritance, the whole chain.

**Editing configuration has no side effects.** No files are deleted — files on disk belong to the
user. The saved handle is *not* cleared; it still serves "Run from here". The only change is a
dashed amber ring. A node can be `done` **and** `stale` at once, and that combination is exactly
the information the user needs.

**Required copy, because it is a real hazard:** prep stages mutate folders **in place**, so a
`done` output is **not a snapshot**. Re-running `tag` over the same folder skips images that
already have their line written unless `overwrite` is set. *Steps write into the folder itself —
re-running is not a rollback.* Stale is a warning, not a cache.

---

## Toolbox contract

**A tool's output is its function's return value. Its `print()` calls are logs.** Nothing is added
to the tool authoring model.

Today `build_runner_source` prints the result and discards it
(the tail of `build_runner_source`). The change is three lines in the generated shim — the
`print` is kept:

```python
result = getattr(mod, {entry_quoted})(**kwargs)
if result is not None:
    print(result)
try:
    (here / "result.json").write_text(json.dumps(result, default=str), encoding="utf-8")
except Exception as e:                       # a completed tool must not fail on its own postlude
    (here / "result.json").write_text(json.dumps({"_unserializable": str(e)}), encoding="utf-8")
```

`default=str` keeps a returned `Path` or `datetime` from blowing up the shim *after* the real work
is done. The guard is `except Exception`, not `(TypeError, ValueError)`: under `default=str` an
object whose `__str__` raises produces neither. Interactive Toolbox runs are unaffected, and the
extra file is a free win for the Toolbox UI.

**`result.json` is mandatory, and its absence is a failure.** A tool that returns nothing writes
`null`; a file that is missing means the shim never reached its postlude — i.e. the tool raised.
Treating "absent" as `None` would silently take the pass-through branch and carry a green
workflow past a crashed tool.

The shim must also emit an exit-code marker, because a detached node has no `Popen` to `wait()`
on after a restart and the existing `jobs._read_exit_code` maps *unknown* to **success**
(`jobs.py:224-236`, `:141-147`). Prep gets this free from `run_stage`
(`rengu_flow/prep/runner.py:305`); the shim prints the same shape:
`tool exits with return code = N`.

### `materialize_run` must copy `tool.py`

The generated shim resolves the user script relative to itself —
`here = Path(__file__).parent; spec_from_file_location("user_tool", here / "tool.py")`
(`toolbox.py`, the generated shim). A node running out of its own `run_dir` therefore needs `tool.py` copied
there, or it dies with `FileNotFoundError`.

And the copy **must be skipped when `run_dir == tool_dir(tool_id)`** — the interactive path —
because `shutil.copyfile(x, x)` raises `SameFileError`. Getting this wrong breaks all of
`/toolbox`, which is precisely the observable change Phase 0 promises not to make.

`uv_run_argv(tool_id)` keeps its signature (tests call it positionally) and delegates to a new
`uv_argv_for_runner(runner: Path)`.

Return value → emitted handle:

| Returns | Emitted handle |
|---|---|
| `str` | that `path`; format/ext inherited from the input |
| `dict` with a `path` key | present keys win, absent keys inherited |
| `None` | **pass-through** of the input (in-place mutation or pure side effect) |
| anything else | the node **fails**: *"A tool used in a workflow must return a folder path, a dict with a 'path' key, or None."* |

**Input injection is by convention, not configuration.** A tool participates in a workflow by
declaring an input named `path` (optionally `caption_format` / `caption_ext`). If it declares
none, it receives no handle and the node passes through. No mapping UI.

**`[toolbox].enabled` gates tool nodes too.** That switch exists because running a tool executes
arbitrary user Python, and it defaults to `false`. A workflow node runs the same code by the same
mechanism, so exempting it would turn Workflows into a way around the gate — authoring is always
allowed, execution never is until the user opts in. A `tool` node on a host with the gate off
fails at launch with the same message `run_tool` raises, naming
`rengu.local.toml → [toolbox].enabled`. The consequence is intended: tool nodes do not run out of
the box, exactly like the Toolbox page they come from.

### Required fix: concurrent tool runs

`toolbox._active` is keyed by `tool_id` (`toolbox.py:235`) and `last_run.json` is one per tool
folder. Two nodes using the same tool — or a workflow running a tool while `/toolbox` runs it —
collide with `RunActiveError` and overwrite each other's record.

`materialize_run(tool_id, kwargs, *, run_dir) -> argv` is extracted so each node runs in its own
directory; `run_tool` calls it with `run_dir=tool_dir(tool_id)` for identical interactive
behaviour. This is **not optional** — the feature is incorrect without it.

---

## The `train` node

**The node fires a run that is already registered. It does not build one.**

Config is a single field: `job_id`. The user picks an existing run from the queue — one they
already configured, pointing at a dataset TOML that already points at the folder the workflow is
treating. **No config is synthesized, no TOML is rewritten, no path is injected.**

This falls out of how prep actually works: `tag`, `caption`, `quality` and `index` all write **in
place**, so the folder a workflow processes is the same folder the registered dataset already
names. The workflow prepares the data; the run consumes it where it lies. Rewriting a dataset path
would only be necessary if the chain moved the data, and by design it does not.

An earlier draft had the node carry a `config_content` + `dataset_content` pair and rewrite the
first `[[directory]].path`. That was wrong twice over: a training TOML contains no `[[directory]]`
at all — it carries `dataset = "path/to/dataset.toml"`, a library ref `rengu-flow-dataset:<id>`,
or a list of them (`rengu_flow_ui/run_staging.py:92-118`) — and even with the indirection followed,
synthesizing a dataset stub means inventing `resolutions` and `frame_buckets` the user never
chose.

### Queueing semantics

The node puts its run **at the front of the pending queue and starts the queue**, so a workflow
that just spent forty minutes tagging is not stuck behind an unrelated run someone queued
yesterday. Both primitives already exist and need no new code:

- `job_queue.bump_pending_after(job_id)` — *"Ensure job_id is first in pending queue"*
  (`job_queue.py:737`), which renumbers the other pending rows rather than colliding on
  `queue_position`.
- `job_queue.enqueue_existing(job_id)` first when the referenced run is still a `new` draft
  (`job_queue.py:557`) — it refuses anything not in state `new`.
- then `try_start_next()`.

**Guards, so jumping the queue cannot wreck other workflows.** Front-of-queue is a privilege, and
two workflows both claiming it would silently reorder each other:

- If a run is already active, the node **does not preempt it** — `try_start_next` is a no-op and
  the run simply waits its turn at the front. Nothing is killed, ever.
- If another workflow already bumped a run that has not started yet, the second node lands its run
  *behind* it, in bump order, rather than displacing it. A workflow may hold at most one
  front-of-queue claim, tracked as `queue_claim` in its state.
  **The claim has to be settled before the bump, not after.** `bump_pending_after` displaces
  unconditionally and is immediately followed by `try_start_next`, so a runner that bumps first and
  restores the previous claimant afterwards has already started the wrong run — deterministically,
  whenever the queue is idle, which is the normal state. The runner therefore drains the queue for
  the standing claimant *before* the second node bumps, then re-reads the claim.
- If the referenced run is already `running`, `stopping`, or terminal, the node fails validation
  with a readable error instead of re-queuing a run that is already underway.

The node still takes **no GPU lease** — the training lane takes one when the run actually starts —
and still marks `done` as soon as the run is queued and the start has been attempted. That is what
fire-and-forget means here, and it is why the card must read `Queued run #123 →` rather than a
bare green check.

---

## UI

New top-level section **Workflows**, added to `router.ts` and to **both** `el-menu` blocks in
`App.vue` (desktop and mobile drawer).

### List — `/workflows`

Built on the existing `LibraryListPage.vue` + `LibrarySortControls` +
`LibraryItemOverflowMenu` + `useLibraryCrudActions`, the same as `DatasetsListView.vue`. Overflow:
Open · Duplicate · Rename · Delete.

`GET /workflows` returns one row per workflow — `id`, `name`, `steps`, `status`, `updated_at` and
**`chain`**, every node's type in list order. `chain` is what the row draws as
`folder → tag → caption`, and it rides along because the route already parses the graph to count
the steps. Without it the list issued a `GET /workflows/{id}` **per row** purely to learn the node
types — an N+1 whose only defence was capping how many rows bothered to ask.

Rename writes through the graph (`PUT /workflows/{id}` with a changed `graph.name`); there is no
rename route, and the `name` this list sorts on is the column that write keeps in step — see
Persistence.

### Editor — `/workflows/:id`

```
┌ ← Workflows │ Re-tag character set ✎ ──── {x} Variables 2 · [ ▸ Run ][▾] · ⋮ ┐
│ ● Idle · last run 2h ago · Result: D:/datasets/aoi                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  ╭────────────────────────────────────────────────────────────╮             │
│  │ ①  ▣  Source folder                              ● idle    │  ⋮          │
│  │       ${dataset_dir}  →  D:/datasets/aoi · sidecar .txt     │             │
│  ╰────────────────────────────────────────────────────────────╯             │
│  │      ┌──────────────── + ────────────────┐  (hover: insert here)          │
│  ╭─┴──────────────────────────────────────────────────────────╮             │
│  │ ②  ▤  Tag                                        ✓ done    │  ⋮          │
│  │       pixai-v0.9 + cl-tagger-1.02 · max 255 · GPU 0         │             │
│  │       → emits the same folder (writes sidecars in place)    │             │
│  ╰────────────────────────────────────────────────────────────╯             │
│  ├╮                                                                          │
│  ╭┴┴─────────────────────────────────────────────────────────╮              │
│  │ ③  ▤  Quality filter          ⟵ from ①            ⟳ 41%   │  ⋮          │
│  │       blur < 80 · quarantine flagged · CPU                  │             │
│  │       → emits the same folder (survivors)                   │             │
│  ╰────────────────────────────────────────────────────────────╯             │
│                       [ + Add step ]                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Non-consecutive links

The hard part of the vertical model. Three layers, all plain CSS:

1. **The default is silent; the exception shouts.** When `from` is the immediately preceding node,
   **no badge is drawn** — the connector already says it. Only jumps get `⟵ from ①` in an accent
   colour. If everything is badged, no badge is read.
2. **A lane rail** in a fixed ~28 px gutter. A consecutive link is a straight segment in lane 0; a
   jump takes an offset lane running from source past the skipped nodes into the target. Lane
   assignment is interval packing in a pure, tested `assignEdgeLanes(nodes)`. Three lanes cover
   any realistic graph; beyond that it collapses to badge-only.
3. **Hover/focus highlight.** Hovering a card highlights its incoming and outgoing edges; hovering
   the `⟵ from ①` badge dims everything except source, edge and target and outlines the source.
   The badge is a `<button>`, so this works from the keyboard, and Enter scrolls to the source. A
   faint `③ reads past this step` legend appears under the skipped node during hover so it does
   not read as an accident.

### Editing

- **Add** — `[ + Add step ]` at the end, plus a hairline `+` that appears on hover between cards.
  Grouped popover: *Source* · *Prepare* (Tag, Caption, Clean, Quality filter, Quality index) ·
  *Tools* (the Toolbox list) · *Training*. A new node's `from` auto-points at its predecessor.
- **Reorder** — `⋮ → Move up / Move down`. **No drag and drop**: no new dependency, keyboard
  accessible, and lists are 3–8 nodes. Moving above one's own source is disabled with the tooltip
  *"③ reads from ②. Move ② up first."*
- **Delete** — unreferenced nodes delete directly. A referenced node prompts once and **splices
  the chain by default** (children inherit the deleted node's `from`), the least surprising repair.

### Node status

| Status | Chip |
|---|---|
| `idle` | ● grey "Not run" |
| `queued` | ◷ grey "Queued" |
| `waiting_gpu` | ◷ amber "Waiting for GPU · held by run #42" |
| `running` | ⟳ blue + slim `el-progress` + % |
| `done` | ✓ green + finish time |
| `failed` | ✕ red + first line of the error |
| `stale` | ◌ dashed amber ring, **combinable with `done`** |
| `disabled` | dimmed card, struck-through title |

### Running

Split button. **`[ ▸ Run ]`** runs, in list order, every enabled node that is not `done`-and-fresh
— i.e. idle + stale + failed + **stopped**. That is the 90 % case.

`stopped` belongs in that set and the point is not academic: `run_stage` returns **0** when
`should_stop()` fired (`rengu_flow/prep/runner.py:286-307` — `report["stopped"]` is a field, not
an exit code). The runner must map *exit 0 with `report["stopped"] == true`* to `stopped`, never
`done`, and must not propagate the handle as a completed step. Otherwise stopping a caption run at
60 % and pressing Run again skips the remaining 40 % and trains on a half-captioned dataset. Prep
stages resume naturally — that is exactly what `requeue_prep_job` relies on today. **`[▾]`** offers *Run all (force re-run every
step)* and *Validate only*. Per node, `⋮` offers *Run from here* and *Run only this step*.

*Run from here* requires the `from` node to have a saved output; otherwise it is blocked with
*"② has no saved output. Start from ① or earlier."* While running, the primary button becomes
`[ ■ Stop ]`, which cancels the active node and leaves completed outputs intact.

The two per-node entries are the same request with one flag:

| Entry | Body of `POST /workflows/{id}/start` | Plan |
|---|---|---|
| *Run from here* | `{"from_node": "n3"}` | `n3` **and its descendants**, filtered to enabled |
| *Run only this step* | `{"from_node": "n3", "only": true}` | `{n3}` |

`only` is what makes the second entry mean what it says. Without it the menu offered *Run only this
step* and then re-ran every step after it — which for a caption stage is hours of work thrown away
to redo a tag stage. With it, the descendants are **neither reset nor re-run**: `n4` keeps its
`done`, its saved handle and its `config_hash`, and the run finishes as soon as `n3` does.

Two rules survive the narrowing. `only` **still requires the ancestors' saved outputs** — one step
in isolation eats the folder the step before it produced, so the same *"② has no saved output"*
refusal applies, just one node later. And `only` **without** `from_node` is refused outright
(*"Run only this step needs a node to run."*) rather than quietly planning the whole workflow,
which is precisely the lie the flag exists to remove.

### The node drawer

`el-drawer`, rtl, 640 px (100 % on mobile), opened by clicking a card, with `?node=n2` in the URL
so a node is linkable. The header is **persistent across tabs** and carries the progress bar — so
progress is visible while editing config, without a fifth tab.

| Tab | Content | Reuses |
|---|---|---|
| **Configure** | `NodeRuntimeFields` (from / needs GPU / wait for queue / device / enabled) then the per-type form | extracted prep stage forms |
| **Input** | the effective handle and which node it came from; **saved** vs **predicted** (amber) when the source has not run; live folder stats | `useDatasetFolderStats`, `PathValidationFeedback` |
| **Output** | emitted handle + the fixed sentence for the type's output rule; the stage `report.json`; the raw return value for tools; a job link for `train` | **`PrepJobSummaryPanel.vue`** as-is |
| **Logs** | live tail, progress, Stop | **`PrepJobLivePanel.vue`** as-is for prep; `ToolboxLogPanel` for tools |

The **Input** tab is the antidote to getting lost among jumps: it is where the user confirms which
folder actually goes in.

**Where the Output tab's report comes from.** `GET /workflows/{id}/nodes/{node_id}/report` serves
the node dir's `report.json` (prep stages) or `result.json` (tools), parsed, as
`{"file": "report.json", "report": …}`. It reads **from disk**, not from `state_json`: the node
dir is already the source of both files, and copying an unbounded per-run document into the one row
every tick rewrites under compare-and-swap would be paying for it on every progress update.
`_complete_node` reads those files to decide the handle and then drops them, so without this route
the tab had nothing to show for either family. The failures are all 404 with the reason spelled
out: a `folder` or `train` node writes neither file (`train`'s job id is in `state_json`), a node
that never ran has not written one yet, and a node killed mid-write left one that will not parse.

Field hint for the queue toggle, per
[documentation-conventions.md](../developer/documentation-conventions.md): *"Runs only when no
other job holds the GPU. Turn it off to start immediately alongside a training run — both will
share VRAM."*

### Reusing `PrepJobFormView.vue` — extract, do not reuse whole, do not rewrite

The 1640-line form is split into `components/prep/`: `PrepCommonFields.vue`, `TagStageForm.vue`,
`CaptionStageForm.vue`, `CleanStageForm.vue`, `QualityStageForm.vue`, and the new
`IndexStageForm.vue`. `buildConfig()` (`PrepJobFormView.vue:1215`) splits into a pure
`buildStageConfig(stage, forms)` in `src/lib/prepStageConfig.ts` with a co-located test.

- **Not reused whole** because the view carries page chrome (page-head, back button, Queue/Start
  buttons, `chainCaption`) that means nothing inside a drawer, and reads `route.params.stage`
  directly. It is a view, not a component.
- **Not rewritten** because the per-model thresholds, the prompt preview against
  `POST /prep/caption-prompts/preview`, and the quality preview run are behaviours with a settled
  server contract. Retyping them duplicates ~700 lines that drift on the first backend change.
- **The cut is already drawn**: `PrepJobSummaryPanel.vue` already takes `:tag-form`,
  `:caption-form`, `:clean-form` and `:quality-form` as separate props.

In a workflow the forms do **not** supply `path` / `caption_format` / `caption_ext`; the executor
injects them from the handle. That omission is the core of the reuse.

Similarly, `ToolboxRunPanel.vue`'s input block and output console are extracted as
`ToolboxInputsForm.vue` and `ToolboxLogPanel.vue`, consumed by both Toolbox and the node drawer.

---

## Prep migration

`rengu_flow_ui/prep_jobs.py:1-8` states the current guarantee outright: prep shares the queue *"so
a prep job never shares the GPU with a training run"*. **Removing prep from that queue removes the
guarantee.** The phase order below is therefore not negotiable.

### Phase 0 — groundwork (mergeable alone)

1. Extract path-based log helpers in `jobs.py` (`tail_log_path`, `read_raw_log_tail_path`,
   `log_tail_start_offset_path`); the `job_id` wrappers delegate. This hands workflows the entire
   log-streaming stack — CRLF decoding, byte-offset tailing, `iter_log_frames` framing, marker
   stripping — for free. The extraction is purely mechanical: none of the three touches
   `JobRecord` state beyond `log_path`. Two contracts must survive verbatim: `tail_log` propagates
   `KeyError` while `log_tail_start_offset` swallows it and returns `0` (`app.py:951` depends on
   that), and `LOG_WS_TAIL_BYTES` must stay defined above any helper that defaults to it.
   `iter_log_frames` is already pure — leave it alone.
2. Extract `toolbox.materialize_run` (copying `tool.py`, guarded against `SameFileError`) and
   `uv_argv_for_runner`; add `result.json` and the exit-code marker to `build_runner_source`.
3. Add `system_stats.list_gpu_devices()` plus the per-process cache. Measured on the dev host:
   **1616 ms cold, ~31 ms warm** — the cache is not an optimization, it is what keeps the tick
   from stalling on a subprocess.
4. Add `gpu_lease.py`, the `gpu_leases` table and `jobs.gpu_index`, **wired to the training lane
   only**: acquire in `try_start_next`, eager release in `poll_job`, `reap_dead()` at the top of
   every `_tick` **and** at the top of `try_start_next`, `reconcile_on_start()` in the app
   lifespan. `_tick` gains its `BaseException` guard and a separate `try/except` per step — a
   failure in `reap_dead` must not take down the run reconciliation that works today.

Every `except` that wraps `jobs.start_job` must be **`BaseException`**, not `Exception`.
`ensure_training_extras` → `ensure_profiles` raises `SystemExit`
(`rengu_flow/install/manager.py:105-110`). Caught only as `Exception`, the release is skipped, the
row stays `pending`, and the lease keeps `pid IS NULL` — which by the no-timer rule is *immortal*.
Every retry then collides with the job's own lease, so fixing the environment does not revive the
queue; only restarting the server does.

**"No observable behaviour change" is a claim that has to be proven, not asserted.** The
structural argument is that starting now requires `not has_active_runner()` **and** `acquire()`,
and in Phase 0 the only possible holder is the running training job — exactly when
`has_active_runner()` is already `True`. The extra condition is therefore redundant in steady
state, which means **every behavioural difference is, by construction, a lease leak**.

Two changes are visible and intended, and the phase description should not pretend otherwise: a
Toolbox run now writes `result.json` into the tool folder and prints a
`tool exits with return code = N` line at the end of its console output.

**A green suite does not prove the claim.** The regression tests that do are the ones the training
lane never had: that `try_start_next` acquires, that `poll_job` releases, that a `start_job` raising
`SystemExit` still releases, that `stop_job` on a `pid is None` row unwedges the queue, and that
`_tick` survives `SystemExit`. Each is ~15 lines, and each fails against a plausible wrong
implementation while 1500 other tests stay green.

One existing test does change: `tests/test_ui_job_queue.py::test_try_start_next_after_finish`
stubs `jobs.poll_job` and flips the state by hand, so the eager release never runs. It passes
unmodified **only because** `reap_dead()` runs at the top of `try_start_next` and frees the lease
of a job whose row is now `finished`. That test is the regression guard for the whole
holder-validity design — if it needs editing, the design is wrong.

### Phase 1 — workflows alongside prep (nothing deleted)

5. `workflow_db.py`, `workflow_graph.py`, `workflow_nodes.py`, `workflow_runner.py`,
   `workflow_routes.py`; the tick hooked into `queue_poller`.
6. The `PrepJobFormView.vue` extraction and the new frontend section.
7. Both routes coexist; the user validates the new path on real datasets with the old one as a
   safety net.

### Phase 2 — removal

8. Delete `prep_jobs.py`; delete `POST /prep/jobs` and `POST /prep/jobs/{id}/requeue`. **Keep**
   `GET /prep/jobs/{id}/report` and `/config` while the history view exists.
9. Remove the `kind == "prep"` branch from `jobs.start_job` (`jobs.py:68-75`); move
   `build_prep_command` (`jobs.py:37`) into `workflow_nodes.py`. `jobs.py` becomes
   training-only.
10. `/prep/new/:stage` becomes a shortcut that creates a one-node workflow and redirects — not a
    404; the URL is bookmarked.
11. Drop `createPrepJob` / `requeuePrepJob` from `api.ts`. `createPrepJob`'s only caller outside
    `PrepJobFormView` is `QualityIndexView.vue:399`; `requeuePrepJob`'s only caller is
    `PrepJobsView.vue:414`. **Order matters**: `requeuePrepJob` cannot leave `api.ts` before step
    12 removes the requeue button, or the build breaks in between.
12. `PrepJobsView.vue` ships one release as read-only "Prep history (legacy)", then is deleted.
13. **Only after** that: remove the `kind == "prep"` check in `live_stream.snapshot_job_live`
    (`:45`) and the `kind == "train"` filters in `training_hub.py` (`:400`, `:440`). Earlier would
    break the history view.

### Existing `kind='prep'` rows

Terminal rows need **nothing**: they are not migrated, not deleted, and the `kind` column is not
touched. `GET /jobs` already defaults to `kind="train"`, so they become invisible with no effort.
This is what keeps `SCHEMA_VERSION` at 3 and avoids a wipe that would destroy the user's dataset
library. The one-line `if job.kind != "prep"` guard in `jobs.poll_job` stays permanently — it is
what lets orphaned prep rows still reconcile, and it costs nothing.

**`pending` rows are a different matter, and "invisible" is not "inert".**
`job_queue._pending_sorted()` (`job_queue.py:108-113`) filters on `state == "pending"` with no
`kind` filter, and step 9 removes the `kind == "prep"` branch from `jobs.start_job`. A user who
upgrades with a queued prep job — the *normal* state, since the `chainCaption` toggle enqueues the
caption job with `start_now: false` — would have `try_start_next` pick that row, fall into the
`else` branch, and launch **DeepSpeed against `prep.toml`**. It burns an `ensure_training_extras`
pass, fails, and blocks the real queue while it does, because `has_active_runner()` is global.

Phase 2 therefore adds two things: `kind == "train"` to `_pending_sorted()` and
`has_active_runner()`, and a one-shot startup sweep marking any non-terminal `kind='prep'` row
`stopped` with an explanatory `error`.

---

## Risks and known limits

1. **🔴 GPU exclusion regression if the phases are reordered.** Shipping the prep migration before
   the lease means a `caption` node (vLLM, ~14 GB) can start on top of a training run. Phase 0
   before Phase 2, always.
2. **🔴 `has_active_runner()` does not filter by `kind`** (`job_queue.py:130`). A `kind='prep'` row
   stuck in `running` — server hard-killed — kills the training queue permanently. It reconciles
   for free through `jobs.refresh_all_jobs` → `poll_job`, **as long as `poll_job` is not touched**.
   This needs an explicit regression test, not an assumption.
3. **🟡 Device selection does not buy parallelism while `has_active_runner()` is global.** With
   `wait: true`, picking GPU 1 still waits for the job on GPU 0. In v1 the field means *"when its
   turn comes, run it on GPU N"*; real parallelism comes from `wait: false`. Per-device slots is a
   `job_queue.py` redesign — see [Open work](#open-work).
4. **🟡 A `train` node's green check does not mean "trained."** `done` means "job enqueued". The
   card must read `Queued run #123 →` with a link to `/runs/jobs/123`, never a bare check.
5. **🟡 Prep outputs are not snapshots.** Every stage mutates folders, so "start from an
   intermediate step" replays against the folder *as it is now*, not as it was when the step ran.
6. **🟡 `quality.output_dir` holds the rejects, not the result.** The most likely naming trap in
   the feature; mitigated by the `clean`-only rule in `collect_output`, the "Quarantine folder"
   relabel, and the per-card output sentence.
7. **🟡 One-node workflows accumulate.** Prep-as-a-one-node-workflow creates a row per one-off job.
   Auto-name them (`Tag — aoi`) and offer *Discard* in the editor, or the list fills with noise
   inside a week.
8. **🟡 Client/server duplication of variable resolution.** ~25 lines exist twice, tested on both
   sides. Accepted over a round-trip per keystroke; if they drift, the server wins. Note this is
   the *only* accepted duplication — staleness hashing is server-side exclusively, for the
   float-serialization reason given above.
9. **🟡 `wait: false` nodes are invisible to `snapshot()`**, so the "GPU 0 — held by …" badge and
   the device picker under-report real usage, and a training job queued five minutes later starts
   on top of the node the user opted out with. The field hint warns about *already running*
   training; it cannot warn about future training.
10. **🟡 The GPU "wait queue" has no fairness.** It is a retry loop, not a queue: a stream of short
    training jobs can starve a `wait: true` node indefinitely. The UI must say "retries
    opportunistically, no ordering guarantee" rather than imply FIFO.
11. **🟡 `rengu_flow/utils/signal_files.py` imports `torch` at module scope** (line 10). Dropping
    `SIGNAL_QUIT` from the tick would pay a multi-second torch import inside the poller thread on
    the first cancel, stalling training reconciliation with it. Import it lazily, as
    `jobs.stop_job` and `prep_jobs` already do, or hard-code the filename in `workflow_nodes`.
12. **🟡 Two `folder` nodes, or a `folder` mid-chain, are legal but unmodelled.** Execution order
    is still list order, the rail draws no fork, and the run bar's single "Result:" silently
    reports only the last node's handle. A newly inserted `folder` must also get `from: null`, not
    the auto-assigned predecessor.
13. **🟡 `clean(in_place=true)` twice is silently non-idempotent.** The second pass re-inpaints
    already-inpainted images and writes a second timestamped "originals" backup whose originals are
    the first pass's *output* — so the thing that looks like a rollback is not one.
14. **🟡 The "prep never shares the GPU with training" guarantee already has a hole today.**
    `prep_jobs.enqueue_prep_job(start_now=True)` and `requeue_prep_job(start_now=True)` call
    `jobs.start_job` directly — no lease, no `_start_lock`, no `try_start_next`. A training job
    sets `state="running"` only *after* `ensure_training_extras`, so throughout that
    minutes-long window the row still reads `pending`, `has_active_runner()` is `False`, and a
    `start_now` prep job walks straight past the guard onto the held GPU. Reproduced. This is
    pre-existing, not a regression of Phase 0 — but it weakens the premise the phase ordering
    rests on, and both call sites must acquire a lease before Phase 2 removes the shared queue.

---

## Files

**New — `rengu_flow_ui/`:** `gpu_lease.py`, `workflow_graph.py` (dataclasses, tolerant parsing,
`validate`, `resolve_variables`, `node_config_hash`, `effective_output`), `workflow_db.py` (CRUD,
`node_dir`, clone), `workflow_nodes.py` (`build_launch` / `run_inline` / `collect_output`),
`workflow_runner.py` (`tick`, `reconcile_on_start`), `workflow_routes.py`
(`register_workflow_routes`, mirroring `prep_routes.py`).

`workflow_nodes.py` uses an `if node_type.startswith("prep.")` chain, **not a registry**. The
repo's registries are for pluggable training components; three node families do not warrant one.

**Modified — `rengu_flow_ui/`:** `queue_poller.py` (`_tick`), `job_queue.py` (acquire),
`jobs.py` (release, `env` from `gpu_index`, log-helper extraction, `kind=='prep'` removal),
`toolbox.py` (`materialize_run`, `result.json`), `library_db.py` (`workflows` table), `db.py`
(`gpu_index` additive column), `app.py` (route registration), `settings.py` (`workflows_dir()`).
**Modified — `rengu_track/`:** `system_stats.py` (`list_gpu_devices`).

**New — `ui/web/src/`:** `views/WorkflowsListView.vue`, `views/WorkflowEditorView.vue`;
`components/workflow/` (`WorkflowNodeCard`, `WorkflowEdgeGutter`, `WorkflowNodeDrawer`,
`WorkflowAddStepPopover`, `WorkflowVariablesDialog`, `WorkflowRunBar`, `nodeforms/*`);
`components/prep/` (the five stage forms + `PrepCommonFields`); `ToolboxInputsForm.vue`,
`ToolboxLogPanel.vue`; `composables/useWorkflowEditor.ts`, `useWorkflowRun.ts`;
`types/workflow.ts`.

Pure logic goes in `src/lib/` with a co-located `.test.ts`, per repo convention:
`workflowGraph.ts` (add/remove/move/re-point, `from` legality, splice-on-delete),
`workflowVars.ts`, `workflowHash.ts` (stable stringify, `computeStale`), `workflowLayout.ts`
(`assignEdgeLanes`), `workflowNodeTypes.ts` (the catalog: label, icon, GPU default,
`describeOutput`), `prepStageConfig.ts`.

**Modified — `ui/web/src/`:** `api.ts`, `router.ts`, `App.vue` (both menus), `PrepJobFormView.vue`
(reduced to a shell), `ToolboxRunPanel.vue`, `QualityIndexView.vue`.

**Tests:** `tests/test_gpu_lease.py` (CAS under contention, reap on dead pid, pid reuse),
`tests/test_workflow_graph.py` (variables, hash chain, `from` invariant, `effective_output` per
type — especially `quality` vs `clean`), `tests/test_workflow_runner.py` (handle propagation with
fake launchers, restart reconciliation, cancel escalation), plus a regression test that a stale
`kind='prep'` row cannot wedge the training queue.

---

## Open work

- Per-device queue slots in `job_queue.py`, so `wait: true` on GPU 1 does not block behind GPU 0.
- On-disk node history (`<node_id>.<timestamp>`) if losing the previous run's `report.json` proves
  painful; a one-line change given `workflow_db.node_dir`.
- A "save to dataset library" node.
- Import/export of a workflow as a JSON file, matching the dataset library's escape hatch.
- Apply `pid_create_time` reconciliation to `jobs.poll_job` as well; the same reboot hazard applies
  to long-lived training rows, just less often.

## References

- [developer/web-ui.md](../developer/web-ui.md) — control-plane module map, job queue, field help
- [user/dataset-prep.md](../user/dataset-prep.md), [user/toolbox.md](../user/toolbox.md)
- [developer/documentation-conventions.md](../developer/documentation-conventions.md) — field hint
  and form-anatomy rules this UI must follow
- `rengu_flow/prep/config.py` — the stage config dataclasses a node's `config` mirrors
- `rengu_flow/control/progress_stream.py` — the `@@RFPROG@@` protocol node progress reuses

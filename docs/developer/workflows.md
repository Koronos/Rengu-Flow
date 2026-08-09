# Workflows (developer guide)

Control-plane feature: a chain of prep / Toolbox / training-launch nodes over one dataset folder,
with one saved state per workflow. Lives entirely in `rengu_flow_ui/` + `ui/web/`; the training core
is untouched.

User guide: **[docs/user/workflows.md](../user/workflows.md)**.
Design rationale — *why* each rule exists, and the alternatives that were rejected:
**[docs/spec/workflows.md](../spec/workflows.md)**. This page is the contract and the map; it does
not repeat the reasoning.

Not shipped yet (see [BACKLOG.md](../BACKLOG.md), P5-3): retiring `kind='prep'`. `/prep`,
`prep_jobs.py` and `POST /prep/jobs` still work exactly as before and coexist with workflows.

## Layout

| Path | Role |
|------|------|
| `rengu_flow_ui/workflow_graph.py` | **Pure model.** Dataclasses (`WorkflowGraph`, `WorkflowNode`, `NodeGpu`, `DatasetHandle`), the `NODE_TYPES` catalog, tolerant parsing, `validate`, `resolve_config`, `materialize_config`, `node_config_hash`, `compute_stale`, `effective_output`. No DB, no filesystem, no subprocess |
| `rengu_flow_ui/workflow_db.py` | **Persistence.** CRUD over the `workflows` row, `update_graph` (optimistic concurrency), `mutate_state` (compare-and-swap), `clone_workflow`, `node_dir`, the `workflows_version()` counter the events WS watches. Treats `content` as an opaque string |
| `rengu_flow_ui/workflow_nodes.py` | **One node's mechanics.** `build_launch` (argv + env), `run_inline` (`folder`, `train`), `collect_output`, `read_exit_code`, `needs_lease` / `lease_devices`. Schedules nothing, writes no state |
| `rengu_flow_ui/workflow_runner.py` | **The scheduler.** `tick`, `start_workflow`, `cancel_workflow`, `reconcile_on_start`; node lifecycle, GPU acquire/release, cancellation escalation, restart adoption. The only writer of run state |
| `rengu_flow_ui/workflow_routes.py` | **HTTP/WS.** `register_workflow_routes(app)`, mirroring `prep_routes.py`. Imports `workflow_runner` lazily inside the two handlers that need it |
| `rengu_flow_ui/gpu_lease.py` | The `gpu_leases` table and its API. Shared by both lanes — not workflow-specific |

Wiring:

| Where | What |
|-------|------|
| `rengu_flow_ui/library_db.py` | `CREATE TABLE IF NOT EXISTS workflows` inside `init_library_tables` — additive, like `datasets`. `SCHEMA_VERSION` does **not** move |
| `rengu_flow_ui/queue_poller.py:_tick` | Three separately guarded steps: `gpu_lease.reap_dead` → `jobs.refresh_all_jobs` → `workflow_runner.tick`. Each in its own `try` catching `BaseException` (`ensure_profiles` raises `SystemExit`), so a failure in the newest step cannot take down run reconciliation |
| `rengu_flow_ui/app.py` lifespan (`:229-240`) | `gpu_lease.reconcile_on_start()` **then** `workflow_runner.reconcile_on_start()` — the order matters: a node whose launch never happened must have lost its unbound lease before the lane looks at it |
| `rengu_flow_ui/app.py:298-302` | `register_workflow_routes(app)` |

Node directories are `ui_data_dir()/workflows/<workflow_id>/<node_id>`, produced by
`workflow_db.node_dir` (path-traversal guarded, mirroring `toolbox.tool_dir`). There is no
`settings.workflows_dir()` — the spec's file list names one, the implementation composes the path in
`node_dir` instead.

## Frontend

| Path | Role |
|------|------|
| `views/WorkflowsListView.vue` | List page. `GET /workflows` returns `chain` + `steps` precomputed, so search/sort are client-side with no per-row hydration |
| `views/WorkflowEditorView.vue` | The editor: renders the card list + gutter, owns hover/jump highlighting and the `?node=` deep link, wires the two composables |
| `components/workflow/WorkflowNodeCard.vue` | One card. Purely presentational — chip, summary, output sentence, jump badge all arrive as props |
| `components/workflow/WorkflowEdgeGutter.vue` | Paints one row's slice of the connector rail from `GutterSegment[]`. CSS only, no canvas, no SVG, no graph library |
| `components/workflow/WorkflowNodeDrawer.vue` | The four-tab drawer (Configure / Input / Output / Logs); predicts an unsaved node's output client-side, polls the node log, fetches the report |
| `components/workflow/WorkflowAddStepPopover.vue` | Grouped add menu from `NODE_TYPE_GROUPS`; the Tools group is seeded from the Toolbox list |
| `components/workflow/WorkflowRunBar.vue` | Header: rename, Run split button, Stop, Variables badge, status line |
| `components/workflow/WorkflowVariablesDialog.vue` | Variable editor plus the "Used by" table from `collectRefs` |
| `components/workflow/nodeforms/` | `NodeRuntimeFields.vue` (from / GPU / device / enabled — shared by every non-`folder` type), `FolderNodeForm.vue`, `ToolNodeForm.vue`, `TrainNodeForm.vue` |
| `components/prep/` | `PrepCommonFields.vue` + the five stage forms (`Tag`, `Caption`, `Clean`, `Quality`, `Index`), extracted from `PrepJobFormView.vue` and consumed by both `/prep` and the drawer |
| `lib/workflowGraph.ts` | Pure immutable graph edits: `createNode`, `addNode`, `removeNode` (splices children onto the deleted node's `from`), `canMove`/`moveNode`, `legalSources`, `repointNode`, `ordinals` |
| `lib/workflowNodeTypes.ts` | The catalog mirroring `workflow_graph.NODE_TYPES`: label, icon, group, `consumes`/`emits`, `defaultNeedsGpu`, and `describeOutput` — the single source for the add menu, the card and the Output tab |
| `lib/workflowStatus.ts` | Status chips and run eligibility (`nodesToRun`, `runFromBlockReason`, `moveBlockReason`). **Renders** `stale`, never computes it |
| `lib/workflowVars.ts` | `${name}` substitution for the inline preview only |
| `lib/workflowLayout.ts`, `lib/workflowGutter.ts` | Lane assignment (interval packing) and per-row gutter geometry for jump connectors |
| `lib/workflowCard.ts` | Card prose: config summary, chain summary, result path, relative time |
| `lib/prepStageConfig.ts` | `buildStageConfig(stage, forms)` — the pure half of the old `PrepJobFormView.buildConfig()` |
| `composables/useWorkflowEditor.ts` | Document state: graph/state/stale/version, dirty tracking, debounced autosave, 409 handling, read-only while running |
| `composables/useWorkflowRun.ts` | Live state: the events WebSocket, reconnect backoff, node-log progress polling, and `start`/`stop`/`validate` |
| `types/workflow.ts` | The wire shapes, annotated with their Python source |

Every `lib/workflow*.ts` and `lib/prepStageConfig.ts` is pure (no Vue import) and has a co-located
`.test.ts`; both composables have tests too.

## The `from` invariant

`from` may only reference a node that appears **earlier in the list**
(`workflow_graph.validate`). Three consequences the code depends on:

- **Cycles are unexpressable**, so there is no cycle detection anywhere.
- **`execution_order` is list order**, filtered to enabled nodes. That is the entire scheduler
  (`workflow_graph.execution_order`, `workflow_runner._next_runnable`).
- **One forward pass computes any closure.** `workflow_runner._config_hashes` builds the hash chain
  and `_with_descendants` builds the "run from here" set in a single loop, because a parent is
  always visited before its children.

`from` decides *which folder* a node reads, never *when* it runs. Non-consecutive links are just a
`from` that skips; `lib/workflowLayout.assignEdgeLanes` handles drawing them.

`from: null` is legal only for `folder` and `tool` (`NodeType.source_optional`). Anything else
without a source is a validation error, not a mid-run failure — deleting a `folder` node splices
its children onto its own `null` source, which is exactly how a sourceless `prep.clean` used to
reach the launcher.

## Persistence: two columns, one row

```sql
CREATE TABLE IF NOT EXISTS workflows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,               -- graph JSON
    state_json TEXT NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

`content` and `state_json` are **never written by the same caller**:

| Column | Sole writer | Concurrency control |
|--------|-------------|---------------------|
| `content` (+ `name`) | `workflow_db.update_graph`, from `PUT /workflows/{id}` | `UPDATE ... WHERE id = ? AND version = ?`; zero rows → `StaleWorkflowError` → HTTP 409 |
| `state_json` | `workflow_db.mutate_state` only | read-modify-write inside `UPDATE ... WHERE state_json = ?` (the previously read raw value), retried on conflict |

That split is what makes "edit node 3, save, and nodes 1–2 keep their `done` and their outputs"
work: an editor save cannot clobber live progress, and a poller tick cannot clobber an edit.

`mutate_state(wf_id, fn)` takes a callback that mutates the dict in place (returning `None`) or
returns a replacement. It retries the whole read-apply-write against the fresh value; without the
CAS, a `/cancel` writing `cancelling` and a poller writing progress would race on a JSON blob and
whichever landed second would win. Three threads reach it routinely: the poller, `/start` and
`/cancel` (FastAPI runs sync endpoints in a threadpool). Workflows are the one object in the app
with **no history**, so a lost update here is unrecoverable.

`name` is a denormalized copy of the graph's own `name`, so the list view can sort and search
without parsing N blobs. There is no rename route — the UI renames by saving a graph with a changed
`name`, and `update_workflow_route` passes `graph.name` to `update_graph` so the column moves in the
same transaction. `POST /clone` rewrites the graph's `name` **before** the row lands
(`workflow_routes.clone_workflow_route`), or the copy would rename itself back on its first save.
Clone discards `state_json` and resets `version` to 0.

`workflow_db._workflows_version` is a monotonic counter bumped on every row write; the
`/workflows/events/ws` endpoint watches it and pushes `{"type": "workflows-changed"}` so clients
refresh without polling. Calqued on `db._jobs_version`.

## `state_json`

```jsonc
{
  "status": "idle | running | cancelling | failed | stopped | done",
  "current_node": "n2",
  "started_at": null, "finished_at": null,
  "nodes": {
    "n2": {
      "status": "pending|waiting_gpu|launching|running|stopping|done|failed|stopped|skipped",
      "pid": 4231, "pid_create_time": 1754689201.4, "exit_code": 0,
      "started_at": "…", "finished_at": "…",
      "output": { "path": "…", "caption_format": "sidecar", "caption_ext": ".txt" },
      "saved_input": { … }, "config_hash": "…", "error": "",
      "adopted": false, "log_size": 0, "stop_requested_at": null, "result": null
    }
  },
  "queue_claim": { "job_id": 42, "node_id": "n5" }
}
```

Three fields carry more weight than they look:

- **`output` is stored resolved**, not as a reference to the upstream node. That is what makes
  "start from step 4" possible without re-running 1–3 (`workflow_runner._input_handle`).
- **`saved_input` is written next to `output`** by `_complete_node`. `compute_stale` compares it;
  omitted, every comparison is `None != handle` and the whole chain reads stale forever.
- **`config_hash` is what "has already run" means** for staleness — not the presence of an output.
  `train` emits nothing, so testing for an output would make a done training node permanently
  fresh.

## Node lifecycle

```
pending ──▶ waiting_gpu ──▶ launching ──▶ running ──▶ done
   │             │                           │    └──▶ failed
   │             │                           └──▶ stopping ──▶ stopped
   └──────── skipped (disabled)
```

`workflow_runner.tick()` takes a **non-blocking** lock (`_tick_lock`, mirroring
`job_queue._start_lock`) and returns immediately if it is held: two concurrent `/start` calls must
not both spawn the same `pending` node, and a CPU-only node has no lease to serialize them.

`tick()` is called from `queue_poller._tick` and synchronously at the end of `/start` and `/cancel`
so the response carries the effect. `start_workflow` and `cancel_workflow` **plan only**; the caller
ticks.

`_advance(workflow_id)` is one pass over one workflow: reconcile `current_node` if it has one,
escalate a pending cancel, then launch nodes in list order. Inline nodes complete synchronously, so
the loop keeps going rather than costing a tick per node.

Rules the state machine will silently break if you edit around them:

| Rule | Where | What goes wrong without it |
|------|-------|----------------------------|
| `launching` is written **before** `popen` | `_mark_launching`, called inside `_launch_node` before `_install_prep_extras` | A server killed in that window leaves the node `pending` while its detached child keeps running; on restart a second `prep.tag` writes the same sidecars |
| A `launching` node is **never auto-started** on reconcile | `_reconcile_workflow`, `_adopt_launching` | Same. It is failed with `LAUNCH_INTERRUPTED_ERROR`, or adopted when `node.log` is measurably growing across two samples 1 s apart |
| `node.log` is **truncated per run**, never appended | `_spawn` | `read_exit_code` returns the last marker *in the file*, so run 2 killed before printing its own would inherit run 1's `= 0` |
| Unknown exit code is **not** `done` | `workflow_nodes.read_exit_code` returns `None`; `_finalize_node` maps it to `EXIT_UNKNOWN_ERROR` | Deliberately unlike `jobs._read_exit_code`, which maps unknown to success. Here that would propagate a handle on no evidence |
| exit 0 **with `report["stopped"] == true`** is `stopped` | `_finalize_node` | `run_stage` returns 0 when `should_stop()` fired. Calling it `done` lets a caption run stopped at 60 % be treated as complete |
| The lease is released under `finally` | `_finalize_node` | Any verdict that raises on the way out would leave the training lane blocked forever |
| Liveness matches `pid` **and** `pid_create_time` | `_pid_is_alive`, `gpu_lease._pid_is_gone` | After a reboot, PID 4231 is an unrelated process; the node stays "running" forever and its lease is never released. Zombies count as dead |

`failed` and `stopped` **halt** the workflow rather than being stepped over (`_next_runnable`): the
chain's premise is that each node's folder is the next node's input.

### Cancellation

Two-phase, because a tick comes back to look again (`_escalate_cancel`):

1. `/cancel` sets `status = "cancelling"` and ticks.
2. First pass: drop `SIGNAL_QUIT` in the node dir, set the node `stopping`, record
   `stop_requested_at`. Prep stages poll the signal between batches via
   `rengu_flow/prep/runner.py::_make_should_stop`, so a half-done tag writes its `report.json`.
3. After `CANCEL_GRACE_SECONDS` (20 s): `terminate_process_tree(pid)`.

Tool nodes have no signal contract by decision and are terminated on the first pass. `train` nodes
have nothing to stop. `_stop_waiting_nodes` moves any `waiting_gpu` node to `stopped` but
**keeps its saved output** — that node never started, so "run from here" must still work.

`_sweep_signal_files` clears stale `quit` / `save_quit` before each launch, as
`prep_jobs.requeue_prep_job` does; without it, re-running a cancelled node exits immediately.
`rengu_flow/utils/signal_files.py` imports `torch` at module scope, so both helpers import it
**lazily** — paying that import at module load would put it inside the poller thread's first pass.

### Restart reconciliation

`workflow_runner.reconcile_on_start()` runs from the app lifespan, after `gpu_lease.reconcile_on_start()`.
Node subprocesses survive a restart (own process group via `popen_repo_subprocess`, stdout to a
file), deliberately unlike Toolbox, which loses its in-memory handle and marks a live process
`failed`. Per workflow (`_reconcile_workflow`):

- `_collect_garbage` drops state rows for nodes the editor deleted and releases their leases.
- `waiting_gpu` → release the lease and reset to `pending`, so the decision is re-made from scratch.
- `launching` → adopt if the log is growing, else fail + release.
- `running` / `stopping` → `_finalize_node` when the process is gone.

## GPU arbitration

`gpu_leases` is an additive table whose `PRIMARY KEY (device)` **is** the mutex: inserting over an
occupied device raises `sqlite3.IntegrityError`. That is an atomic compare-and-swap that works
across processes and survives a restart — which in-process memory cannot, since a prep node outlives
the server that spawned it.

```python
enumerate_devices() -> list[int]        # cached per process; -1 when nothing enumerates
acquire(holder_kind, holder_id, devices) -> bool     # devices=None -> every device, one transaction
bind_pid(holder_id, pid, pid_create_time=None) -> bool | None   # False = reaped underneath you
release(holder_id) -> None
reap_dead() -> list[str]
reconcile_on_start() -> list[str]
snapshot() -> list[dict]
wait_reason(devices) -> str
```

Holder ids: `job:<id>` for the training lane, `wf:<workflow_id>:<node_id>` for the workflow lane
(`workflow_runner._holder_id`).

**Reaping is by holder validity, never by a timer** (`gpu_lease.reap_dead`):

- A `job:` lease is freed when the row is missing or its state is not in the allow-list of states in
  which a lease is legitimately held. That allow-list is deliberate: a job state added later
  defaults to *reap*, never to *leak*. Sprinkling `release()` across the five code paths that reach
  a terminal state would be a checklist that goes wrong the first time a sixth is added.
- Any lease whose bound pid is dead, a zombie, or reused (create-time mismatch) is freed.
- There is **no `pid IS NULL` timeout**. The window between `acquire` and `bind_pid` contains a
  `uv sync` that takes minutes on a cold extra. `reconcile_on_start` frees unbound leases instead —
  an unbound lease cannot survive the process that created it.

Both lanes participate. Training acquires inside `job_queue.try_start_next`'s `_start_lock`, after
the `config_path` validation and before `jobs.start_job`; `jobs.poll_job` releases eagerly at its
terminal transition (a latency optimization — `reap_dead` is the correctness mechanism). The
workflow lane acquires in `_launch_node` and releases in `_finalize_node`'s `finally`.

`workflow_nodes.needs_lease(node)` is `gpu.required and gpu.wait` — one definition, consulted by the
runner *before* `build_launch`, because a node that cannot get its GPU is never built.
`lease_devices(node)` returns `None` (auto / host-exclusive) or `[device]`.

### The prep-extras interlock

`_install_prep_extras` runs `ensure_profiles(["prep"])` **in the UI process, with the node in
`launching`** — not inside the child — and refuses **only when the extras are actually missing and
the training lane is busy**. The hazard is the `uv sync` write to `site-packages` under a live
DeepSpeed process, not the node itself; refusing unconditionally would block every CPU-only prep
node during any training run, which is the opposite of the point.

"Busy" is read from `gpu_lease.snapshot()` for a `holder_kind == "train"` holder, **not** from
`job_queue.has_active_runner()` (`workflow_runner._training_lane_busy`). `try_start_next` acquires
the lease and only then calls `ensure_training_extras`, so for that whole multi-minute window the
job row still says `pending`. The row check is kept as a second witness for a run whose lease was
reaped.

A node refused this way raises `_WaitingForLane`, **releases its lease** and returns to
`waiting_gpu` with the reason — holding a GPU token while waiting on a dependency install would
deadlock the lane it is queuing behind.

### Node environment

`workflow_nodes._node_env`: inherited `os.environ` + `PYTHONUNBUFFERED=1` +
`CUDA_VISIBLE_DEVICES` when `gpu.device` is set, merged last. Deliberately **not**
`training_subprocess_env()` — NCCL / TF32 / allocator knobs are training concerns prep only inherits
today because `jobs.start_job` is shared. With `CUDA_VISIBLE_DEVICES=3` the child sees that GPU as
`cuda:0` and all of `rengu_flow/prep/` needs no change.

Per-node device selection is wired end to end. The **training** side is not: `jobs.gpu_index` exists
on `JobRecord` and `job_queue._devices_for_job` reads it, but nothing writes it, so it is always
`None` (host-exclusive). See [BACKLOG.md](../BACKLOG.md) P5-3.

## Output contract per node type — normative

`workflow_graph.effective_output(node, input_handle, report)` owns the rule;
`workflow_nodes.collect_output` only decides what to feed it (a prep stage's `report.json`, a tool's
`result.json`, or nothing). `node.config` is read **literally**, so callers pass a node whose
variables are already resolved (`workflow_runner._resolved`).

| Type | `report` fed | Emits |
|------|--------------|-------|
| `folder` | — | its own `config.path` (+ `caption_format` / `caption_ext` from config). Runs inline; `done` once the directory exists |
| `prep.tag` | `report.json` (ignored) | **the input handle** — sidecars written in place |
| `prep.caption` | `report.json` (ignored) | **the input handle** |
| `prep.quality` | `report.json` (ignored) | **the input handle** — the survivors |
| `prep.index` | `report.json` (ignored) | **the input handle** — the SQLite index lives under `prep_storage_dir()`, outside the dataset |
| `prep.clean` | `report.json` | `in_place ? input : report["output_dir"] or config.output_dir or <input>/cleaned` |
| `tool` | `result.json` | `str` → that path; `dict` with `path` → present keys win, absent inherited; `None` → **pass-through**; anything else → `NodeOutputError(TOOL_RETURN_ERROR)` |
| `train` | — | `None`. Terminal, fire-and-forget |

**The `output_dir` trap.** `cleanup.clean_folder` sets `report["output_dir"]` to the *result* folder
(`rengu_flow/prep/cleanup.py:270-272`); `quality.filter_folder` sets it to the *quarantine* folder,
and only when `action == "move"` (`rengu_flow/prep/quality.py:176-190`). A generic
"read `report['output_dir']`" would make `quality → caption` caption the reject pile.
**`effective_output` consults `output_dir` for `clean` only.** Nothing in `rengu_flow/prep/` changes.
The spec also calls for relabelling the quality field to *Quarantine folder*; that was not done —
`components/prep/QualityStageForm.vue:130-138` still labels it *Output directory*, with a hint
naming it as the destination for flagged images.

**A missing `result.json` is a failure, not a `None`** (`workflow_nodes.collect_output`,
`RESULT_MISSING_ERROR`). The Toolbox shim writes it in its postlude on every successful exit, so an
absent file means the tool raised. Reading it as `None` would take the pass-through branch and carry
a green workflow past a crashed tool. A corrupt prep `report.json`, by contrast, is only logged —
`clean` is the only stage that reads anything out of it, and its fallback is the `output_dir` the
config already names.

## Staleness

`node_config_hash(node, parent_hash, variables)` digests
`(HASH_VERSION, type, materialized config, gpu, parent_hash)`. Because the parent's hash feeds the
child's, **downstream invalidation is free — there is no propagation code**.

```
stale(n) = n has a config_hash (or an output)
           && (current_hash(n) != saved_hash(n) || effective_input(n) != saved_input(n))
```

Two design points to preserve:

- **`materialize_config` hashes the config with defaults filled from the current app version**, not
  the stored partial dict (`workflow_graph.materialize_config`, which round-trips `prep.*` configs
  through `parse_prep_config`). Hashing the partial dict would turn every saved node in every
  workflow amber the day a form gains a field. `HASH_VERSION` covers the inverse case — a changed
  *default* silently altering behaviour while the hash holds still.
- **The hash is computed on the server only.** `compute_stale` runs in `_workflow_detail` and the
  client renders `WorkflowDetail.stale`; there is deliberately no `workflowHash.ts`.
  `QualityStageConfig.blur_threshold` defaults to `80.0`, and `JSON.stringify` emits `80` where
  `json.dumps` emits `80.0` — a client recomputation would disagree on every `prep.quality` node.
  `workflowVars.ts` is the **only** accepted client/server duplication, and only for the inline
  preview; the server decides whether a workflow may run.

Editing configuration has no side effects: no files are deleted and the saved handle is kept so
"Run from here" still works. `done` **and** `stale` at once is a valid, intended combination.

There is no "Accept current configuration" bulk action yet (spec, Staleness) —
see [BACKLOG.md](../BACKLOG.md).

## Run planning

`workflow_runner._plan(graph, state, from_node=, force=, only=)` returns the set of node ids to
reset to `pending`:

| Request | Body of `POST /workflows/{id}/start` | Plan |
|---------|--------------------------------------|------|
| Run | `{}` | every enabled node that is not `done`, or that is stale — i.e. idle + stale + failed + **stopped** |
| Run all (force) | `{"force": true}` | every enabled node |
| Run from here | `{"from_node": "n3"}` | `n3` and its descendants, ∩ enabled |
| Run only this step | `{"from_node": "n3", "only": true}` | `{n3}` — descendants are neither reset nor re-run |

`stopped` is in the default set because prep stages resume naturally; skipping it is what would
train on a dataset captioned only up to 60 %. `only` without `from_node` is refused outright.
`_require_saved_ancestors` applies to both per-node entries and raises with the ordinal glyph
(`①②③…`) the editor numbers cards with, so the error names what the user sees.

`start_workflow` runs `workflow_graph.validate(graph)` first and raises with **every** error joined
by newlines. `validate` is structural *and* substantive: each enabled `prep.*` node is materialized
and put through `PrepConfig.validate_for_stage(stage)` — the same gate the launch runs
(`workflow_graph._prep_config_errors`). Two wrinkles: `validate_for_stage` demands an existing
`path` that in a workflow arrives from the edge, so `_preflight_path()` injects `Path.cwd()` and
path rules are left to the launch; and a node with an unresolved variable is skipped, because the
materialized config is not the one that would run.

## The `train` node

`workflow_nodes._run_train` fires a run that is **already registered**; it never builds one. Config
is a single `job_id`. Prep writes in place, so the folder the workflow processed is the folder the
registered dataset already names — nothing to synthesize, rewrite or inject.

Sequence: `job_queue.enqueue_existing(job.id)` when the run is still a `new` draft →
`job_queue.bump_pending_after(job.id)` → `job_queue.try_start_next()`. Any other state fails
validation. The node takes **no GPU lease** (the training lane takes one when the run starts) and is
`done` as soon as the run is enqueued — which is why the card must read `Queued run #123`, never a
bare check.

`bump_pending_after` displaces unconditionally, so a workflow may hold at most one front-of-queue
claim, tracked as `queue_claim` in `state_json`. **The claim is settled before the bump, not after**
(`_claim_before_bump`, called from `_run_inline_node` *before* `run_inline`): the bump is
immediately followed by `try_start_next`, so restoring a displaced claimant afterwards would already
have started the wrong run.

## API

All routes under `/api/v1`.

| Route | Purpose |
|-------|---------|
| `GET /workflows` | List rows: `id`, `name`, `chain` (node types in list order), `steps`, `status`, `updated_at`. `chain` rides along because the route already parses the graph — without it the list issued one `GET /workflows/{id}` per row |
| `POST /workflows` | Create an empty workflow |
| `GET /workflows/{id}` | `_workflow_detail`: graph + state + per-node `stale` + `version` — the editor's single load payload |
| `PUT /workflows/{id}` | Save the graph. 409 on version mismatch, **and 409 outright while the runner owns the workflow** (`_reject_while_running`) |
| `DELETE /workflows/{id}` | Same running guard: a deleted running workflow would leave a leased, detached child with nothing to reconcile against |
| `POST /workflows/{id}/clone` | Copy `content`, discard `state_json`, rewrite the graph's `name` |
| `POST /workflows/{id}/validate` | `{"errors": [...]}`, runs nothing |
| `POST /workflows/{id}/start` | Body `{from_node?, force?, only?}`. Plans, then ticks synchronously |
| `POST /workflows/{id}/cancel` | Sets `cancelling`, then ticks |
| `GET /workflows/{id}/nodes/{nid}/log` | Byte-offset tail + last `@@RFPROG@@` progress marker |
| `GET /workflows/{id}/nodes/{nid}/report` | `report.json` (prep) or `result.json` (tool), read **from disk**. 404 with the reason for `folder`/`train`, never-run, and unparseable |
| `WS /workflows/{id}/nodes/{nid}/log/ws` | Live tail; closes once the node leaves `running`/`stopping` |
| `WS /workflows/events/ws` | `{"type": "workflows-changed", "version": N}` on every row write |

The log routes reuse `jobs.tail_log_path` / `read_raw_log_tail_path` / `log_tail_start_offset_path` /
`iter_log_frames` — the path-based helpers extracted in P5-1 — so workflows get CRLF decoding,
byte-offset tailing, frame splitting and marker stripping for free.

## Adding a node type

`workflow_nodes` dispatches on an `if node.type.startswith("prep.")` chain, **not a registry**: the
repo's registries exist for pluggable *training* components, and three node families do not warrant
one. To add a type:

1. **`workflow_graph.NODE_TYPES`** — add a `NodeType(type, label, consumes, emits, needs_gpu,
   source_optional=…)`. If the GPU default depends on config, special-case it in
   `default_needs_gpu`.
2. **`workflow_graph.effective_output`** — add the branch. This is the normative rule; get the
   input-vs-new-folder question right before writing any launcher.
3. **`workflow_nodes.build_launch`** — return a `NodeLaunch(argv, env)`, or add the type to
   `INLINE_TYPES` and give it a branch in `run_inline` if it runs in the UI process. Validate
   everything that can fail *before* writing or spawning anything: a bad config must become a node
   `error`, not a traceback in `node.log`.
4. **`workflow_nodes.collect_output`** — decide which file (if any) feeds `effective_output`. If the
   type writes a structured result, add it to `workflow_routes._report_name` so the Output tab can
   serve it.
5. **Exit marker.** A non-inline child must print `… exits with return code = N` as its last line
   (`_EXIT_CODE_RE`). Prep gets it from `run_stage`; the Toolbox shim prints its own. Without it the
   node is failed with `EXIT_UNKNOWN_ERROR` — unknown is never treated as success.
6. **`lib/workflowNodeTypes.ts`** — mirror the catalog entry and add a `describeOutput` sentence.
   Add a form under `components/workflow/nodeforms/` and wire it in `WorkflowNodeDrawer.vue`, and
   seed its form defaults in `lib/workflowGraph.defaultNodeConfig` — **a node is born with the
   form's defaults, never an empty config**, or the card and the run disagree.
7. **Tests**: the output rule in `tests/test_workflow_graph.py`, the launcher in
   `tests/test_workflow_nodes.py`, handle propagation in `tests/test_workflow_runner.py`.

An unknown node type is **preserved on parse and save** and fatal only at execution
(`parse_graph`, `validate`, `build_launch`), so downgrading the app never destroys a graph written
by a newer one. The client mirrors that: `consumesInput` / `emitsHandle` default to `true` for
unknown types so the links around them survive a round-trip.

## Tests

| File | Covers |
|------|--------|
| `tests/test_gpu_lease.py` | CAS under contention, reap on dead pid, pid reuse |
| `tests/test_gpu_lease_wiring.py` | Both lanes: acquire in `try_start_next`, release in `poll_job`, `SystemExit` from `start_job`, `stop_job` on a pid-less row, `_tick` surviving `SystemExit` |
| `tests/test_workflow_graph.py` | Tolerant parsing, variables, the hash chain, the `from` invariant, `effective_output` per type — especially `quality` vs `clean` |
| `tests/test_workflow_db.py` | `update_graph` CAS, `mutate_state` retries, clone, `node_dir` traversal guard |
| `tests/test_workflow_nodes.py` | Launch construction, handle injection, `collect_output`, exit-code parsing |
| `tests/test_workflow_runner.py` | Handle propagation with fake launchers, restart reconciliation, cancel escalation |
| `tests/test_workflow_routes.py` | Route contracts, the 409 guards |
| `ui/web/src/lib/workflow*.test.ts`, `composables/useWorkflow*.test.ts` | The pure graph/layout/status logic and the two composables |

See [testing.md](testing.md) for how to run them.

## References

- [docs/spec/workflows.md](../spec/workflows.md) — the design and the rejected alternatives.
- [docs/user/workflows.md](../user/workflows.md) — the user-facing behaviour this contract produces.
- [web-ui.md](web-ui.md) — the control plane this extends: job queue, staging, field help, live log stack.
- [documentation-conventions.md](documentation-conventions.md) — field-hint and form-anatomy rules the node forms follow.
- `rengu_flow/prep/config.py` — the stage dataclasses a node's `config` mirrors, minus `path` / `caption_format` / `caption_ext`.
- `rengu_flow/control/progress_stream.py` — the `@@RFPROG@@` protocol node progress reuses.

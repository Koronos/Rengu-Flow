# Workflows

A **workflow** is a saved chain of steps that act on one dataset folder: clean it, filter it,
tag it, caption it, run a Toolbox script over it, and fire a training run at the end. Each step's
output folder is the next step's input, and the chain remembers what it produced — so you can
re-run it from step 4 without redoing steps 1–3, or point the whole thing at a different folder by
editing one field.

Workflows live in the **Workflows** section of the web UI (`/workflows`). They do not replace
[Dataset Studio](dataset-prep.md): `/prep` still works exactly as before, and a workflow's prep
steps run the same engine with the same options.

## When to use a workflow

| Use | Reach for |
|-----|-----------|
| One tagging pass over one folder, right now | [Dataset Studio](dataset-prep.md) (`/prep`) |
| The same sequence of steps, repeated on new folders | A workflow |
| Chaining clean → quality → tag → caption without re-typing configuration | A workflow |
| Running a CPU-only step while a training run holds the GPU | A workflow (see [GPU](#gpu-needs-gpu-waiting-and-device)) |
| A custom Python script that feeds a folder into the next step | A workflow with a Tool step |

## What travels between steps

Exactly one thing: a **folder**, plus the caption layout to read and write in it.

| Field | Meaning |
|-------|---------|
| Folder path | The dataset directory the step works on |
| Caption format | `sidecar` (one `.txt` per image) or `json` (one `captions.json` per folder) |
| Caption extension | The sidecar extension, e.g. `.txt` |

Those three are set **once**, on the Source folder step, and inherited down the chain. Individual
prep steps do not carry their own path — that is what makes "process a different folder" a
one-field edit. See [Caption layout](dataset-prep.md#caption-layout) for what the two formats mean.

## Build a workflow

1. Open **Workflows** and click **New workflow**. It is created as *New workflow* and opens the
   editor straight away; rename it with the pencil next to the title.
2. Click **Add step**. The menu is grouped **Source · Prepare · Tools · Training**. Start with
   **Source folder**.
3. Click the card to open its drawer, and fill in the folder path on the **Configure** tab.
4. Add the next step. It automatically reads from the step above it.
5. Repeat. A hairline **+** appears between cards on hover if you want to insert in the middle.

Every step's card shows a one-line summary of its configuration and a fixed sentence saying what
folder it emits. The drawer has four tabs: **Configure**, **Input** (which folder actually goes
in, and where it came from), **Output** (what it emitted, plus the step's report), and **Logs**.

Reordering is `⋮ → Move up / Move down`; there is no drag and drop. Moving a step above its own
source is refused, with a message naming the step you would have to move first.

Deleting a step that others read from splices the chain: those steps inherit the deleted step's
source.

## The step types, and what folder each one emits

This is the part worth reading twice. Most prep stages **write into the folder they are given**
and hand that same folder on. Only `Clean` can produce a new folder.

| Step | What it does | Folder it emits |
|------|--------------|-----------------|
| **Source folder** | Names the dataset folder and its caption layout. No process runs; the step is done as soon as the path exists. | Its configured folder |
| **Tag** | Writes tag sidecars **in place** | The input folder, unchanged |
| **Caption** | Writes caption lines **in place** | The input folder, unchanged |
| **Clean** | Watermark/text removal | **In-place** on → the input folder. Off → the **Output directory** you set, or `<input>/cleaned` |
| **Quality filter** | Scores images and flags the bad ones; when set to move them, the flagged ones go to the quarantine directory | The **input** folder — the survivors |
| **Quality index** | Builds a SQLite score index stored outside the dataset | The input folder, unchanged |
| **Tool** | Runs one of your [Toolbox](toolbox.md) scripts | Whatever the function returns (see [Tool steps](#tool-steps)) |
| **Training run** | Fires an already-registered run | Nothing — training ends the chain |

**The quality-filter trap.** Quality filter's **Output directory** is where *flagged* images are
moved (default `<path>/low_quality`) — it is the reject pile, not the result. A workflow honours
that: `Quality filter` emits the folder it was given, so a `Quality filter → Caption` chain captions
the survivors. Clean is the only stage whose Output directory names the folder the chain continues
with.

**Clean with "Copy undetected images" off** emits a folder containing *only* the images it cleaned;
images where nothing was detected are left behind in the input folder.

## Connecting a step to a non-consecutive one

A step reads from whatever the **From** picker in its drawer names. By default that is the step
immediately above, and no badge is drawn — the connector says it already.

Point **From** at an earlier step and the card grows a `⟵ from ①` badge in the connector gutter,
with a rail drawn past the steps it skips. Hover the badge to highlight source, edge and target.

Two rules:

- **A step may only read from a step above it.** Forward references are refused, which is why a
  workflow can never contain a loop.
- **`From` decides which folder a step reads, never when it runs.** Execution is always top to
  bottom, in list order, one step at a time.

Only Source folder and Tool steps may have no source at all. Every other type needs one, and a
workflow with a sourceless step refuses to start rather than dying half-way through.

## Re-running the same configuration on another folder

This is what workflows exist for. Two ways to do it:

**Edit the Source folder step.** Open its drawer, change the path, save. Every step below it turns
amber (stale), press **Run**, done.

**Use a variable.** Click **{x} Variables** in the header and define e.g. `dataset_dir`. In the
Source folder's path field write `${dataset_dir}`. Now the folder is one field in one dialog, and
the chain of steps never mentions a path at all.

Either way you edit exactly one value. Everything else — tagger models, thresholds, caption
prompt, quality metric — stays as you configured it.

To keep the original and branch off, use `⋮ → Duplicate` in the header. A duplicate copies the
steps and **discards the run state**: it starts with nothing done, so it can never claim another
workflow's outputs as its own.

### Variables

| Rule | Detail |
|------|--------|
| Syntax | `${name}`, where `name` is letters, digits and underscores, not starting with a digit |
| Escape | `$$` produces a literal `$` |
| Where they work | **String fields only** — paths, model ids, prompts, output folders, and `text`/`textarea`/`select` tool inputs. Never numbers, never switches |
| Resolution | One pass, no recursion: a variable whose value contains `${other}` is left as written |
| Undefined name | The workflow **refuses to start** and every offending field is listed at once. It is never substituted with an empty string |

The variables dialog has a **Used by** column listing every field that references each name, and
flags names that are referenced but never defined.

## Running a workflow

The header carries a split button:

| Action | What it runs |
|--------|--------------|
| **▸ Run N** | Every enabled step that is not already done-and-up-to-date: never run, amber (stale), failed, and stopped. `N` is the count |
| **Run all (force)** | Every enabled step, including the ones that are done and fresh |
| **Validate only** | Checks the whole graph and lists every problem. Runs nothing |
| **■ Stop** | Replaces Run while a workflow is running |

Per step, `⋮` offers two more:

| Action | What it runs |
|--------|--------------|
| **Run from here** | That step **and everything below it that reads from it** |
| **Run only this step** | That step alone. The steps below keep their `done` status, their saved folder and their configuration — nothing after it is re-run |

Both require the steps *above* to have a saved output — a single step still eats the folder the
step before it produced. If they do not, the action is blocked with a message naming the earliest
step you have to start from instead.

**Validation happens before anything runs.** Starting a workflow puts every enabled prep step
through the same configuration check the launcher performs, and reports **every** error at once:
a step with no source, a forward `from`, an undefined variable, a Tag step with no models selected,
a Quality index step with no models. You see them all up front, not after forty minutes of tagging.

Folder existence is the one thing pre-flight cannot judge, because a step's folder comes from the
step above and may not exist yet when you press Run. It is checked at launch instead — and since
the Source folder step runs first, a bad source path fails on step 1, before anything is written.

**Stop is graceful first.** The active prep step is asked to stop between batches, so a half-done
tag run finishes its current batch and writes its report instead of losing the work. If it has not
exited after 20 seconds it is killed. Tool steps have no such contract and are killed immediately —
if your script needs cleanup, use `try/finally`. Steps that already finished keep their outputs.

A stopped prep step is *resumable*: pressing Run again continues where it left off, because
already-processed images are skipped (unless Overwrite is on).

### What the cards say

| Chip | Meaning |
|------|---------|
| ● Not run | No saved state for this step |
| ◷ Queued | Waiting its turn in a running workflow |
| ◷ Waiting for GPU | Asked for a GPU lease and did not get it; the second line names who holds it |
| ◷ Launching / ⟳ Running | Started; the card shows a progress bar |
| ✓ Done | Finished, with the finish time (or `Queued run #123` for a training step) |
| ✕ Failed | The first line of the error; the workflow stops here |
| ■ Stopped | Stopped part-way; running again resumes it |
| — Disabled / Skipped | Excluded from this run |
| Dashed amber ring | **Stale** — see below. Combines with any of the above, including ✓ Done |

A failed or stopped step **halts the workflow** rather than being stepped over: everything below it
depends on the folder it was supposed to produce.

### Stale (the amber ring)

A step goes amber when it has run before and its configuration — or the folder it was given —
no longer matches what produced its saved output. Changing a variable, editing a step's form, or
changing an upstream step all turn the affected steps amber, because each step's fingerprint
includes its source's.

Editing configuration has **no side effects**: nothing on disk is deleted, and the saved folder is
kept so **Run from here** still works. The only change is the ring.

> **Re-running is not a rollback.** Tag, Caption, Quality index and in-place Clean write **into the
> folder itself**. A `done` step is not a snapshot of anything, and "start again from step 3"
> replays against the folder as it is now — not as it was when step 3 last ran. Amber is a warning
> that the output no longer matches the configuration, not an undo button. In particular, running
> an in-place Clean twice re-processes already-cleaned images and writes a second "originals"
> backup whose originals are the first pass's output.

## GPU: needs GPU, waiting, and device

Every step except Source folder carries **Needs GPU** in its drawer; the other two controls appear
only while it is on.

| Control | Values | Default | Effect |
|---------|--------|---------|--------|
| **Needs GPU** | on / off | On for Tag, Caption, Clean, Quality index. For Quality filter it follows the metric: off for `blur`, on for `aesthetic` and `iqa`. Off for Tool and Training run | Off means the step never asks for a GPU lease, so it runs alongside a training run |
| **Wait for the GPU queue** | on / off | on | On: the step runs only when nothing else holds the GPU. Off: it starts immediately, sharing VRAM with whatever is already running |
| **Device** | Auto, or a listed GPU | Auto | Sets `CUDA_VISIBLE_DEVICES` for that step only |

Turning **Wait for the GPU queue** off is the explicit escape hatch for "I know these two fit on
one card". Nothing checks that they do; you own the out-of-memory risk. It also makes the step
invisible to the GPU display, so a training run started five minutes later can land on top of it.

Waiting is a retry loop, not a queue: there is **no ordering guarantee**, so a stream of short
training runs can keep a waiting step waiting indefinitely.

Picking a specific device does not buy you parallelism on its own — with **Wait for the GPU queue**
on, a step pinned to GPU 1 still waits for the run holding GPU 0, because the app runs one training
job at a time. Real parallelism comes from turning the wait off.

If a prep step is the first thing on the machine to need the prep dependencies, the workflow
installs them before launching that step, and postpones it while a training run is active — the
install rewrites the shared Python environment, which is not safe under a live run. The step reads
**Waiting for GPU** with that reason until the training queue is idle.

## Tool steps

A Tool step runs one of your [Toolbox](toolbox.md) scripts as part of the chain.

**Execution must be enabled.** Tool steps are gated by the same switch as the Toolbox page, and it
is off by default:

```toml
[toolbox]
enabled = true   # default: false
```

Without it, a Tool step fails at launch naming `rengu.local.toml → [toolbox].enabled`. Authoring is
always allowed; execution is not.

**Your tool joins the chain by declaring an input named `path`.** No mapping UI, no configuration:

| Declared input | Receives |
|----------------|----------|
| `path` | The incoming folder |
| `caption_format` | The incoming caption format (only if `path` is also declared) |
| `caption_ext` | The incoming caption extension (only if `path` is also declared) |

Those parameters are hidden in the step's form — they come from the edge, not from you. A tool that
declares none of them simply receives no folder, and the step passes its input through.

**The tool's return value is the folder it emits.** Its `print()` calls are logs.

| Your function returns | The step emits |
|-----------------------|----------------|
| A string | That folder; caption format and extension inherited from the input |
| A dict with a `path` key | That folder; `caption_format` / `caption_ext` from the dict if present, inherited otherwise |
| `None` | The input folder unchanged — for in-place work or pure side effects |
| Anything else | The step **fails**: *"A tool used in a workflow must return a folder path, a dict with a 'path' key, or None."* |

A tool that raises before returning fails the step; a green workflow can never walk past a crashed
tool.

Each Tool step runs in its own directory, so two steps using the same tool — or a workflow running
a tool while you run it from the Toolbox page — never overwrite each other's inputs or results.

## The Training run step

**The step fires a run you already registered. It does not build one.**

Its only setting is which run to fire, picked from your existing queue. No config is synthesized, no
TOML is rewritten, no path is injected — because prep writes in place, the folder the workflow
prepared is the folder your registered dataset already names.

When the step runs it moves that run to the **front of the pending queue** and starts the queue, so
forty minutes of tagging is not stuck behind an unrelated run someone queued yesterday. It never
preempts anything: if a run is already active, yours simply waits its turn at the front.

> **A green check on a Training run step means "queued", not "trained".** The card reads
> `Queued run #123`, and the drawer's **Output** tab links straight to that run. The step is done
> the moment the run is enqueued; everything after that happens in the training queue, and that is
> where you stop it.

The referenced run must be a saved draft or already queued. If it is running, stopping, or already
finished, the step fails validation instead of re-queuing something that is underway.

## What each step leaves behind

Every step that launches a process gets its own directory under the UI data folder, holding its log
and — depending on the type — the stage report or the tool's return value. Two tabs in the drawer
read them:

- **Logs**: the live tail while the step runs, with progress, and the full output afterwards.
- **Output**: the emitted folder, the sentence describing the type's output rule, and the step's
  structured report (`report.json` for prep stages, the return value for tools).

Source folder and Training run steps run inside the UI process, so they write neither a log nor a
report — a Training run's job id is on its card instead.

## Limits worth knowing before you build something big

- **One saved state per workflow, and no history.** Re-running overwrites the previous result, its
  log and its report. This is a deliberate difference from runs and jobs, which do keep history.
- **One workflow runs at a time.** Starting a second one is refused while another is active.
- **Steps run one at a time, top to bottom**, even when two branches are independent.
- **You cannot edit or delete a workflow while it is running.** The editor goes read-only and the
  server rejects the write; stop the run first.
- **No branching, conditions or loops**, and no draggable canvas — the chain is a vertical list.
- **A second Source folder step mid-chain is legal but unmodelled.** The `Result:` line at the top
  reports one folder: the last enabled step that saved one.
- **The step outputs are not snapshots.** See the warning under [Stale](#stale-the-amber-ring).

## See also

- [Dataset Studio (`rengu prep`)](dataset-prep.md) — what each prep stage actually does, and every
  option in its form.
- [Toolbox](toolbox.md) — authoring the scripts a Tool step runs.
- [Web UI](web-ui.md) — the rest of the control plane: runs, queue, signals, host metrics.

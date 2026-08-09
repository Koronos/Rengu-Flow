import { computed, onBeforeUnmount, ref, watch, type Ref } from "vue";
import { ApiError, api } from "../api";
import { formatError } from "../lib/formatError";
import { isBusy } from "../lib/workflowStatus";
import type { WorkflowDetail, WorkflowGraph, WorkflowState } from "../types/workflow";

/** Long enough that typing a folder path is one save, short enough to feel automatic. */
const SAVE_DEBOUNCE_MS = 700;

export function emptyGraph(): WorkflowGraph {
  return { version: 1, name: "", description: "", variables: [], nodes: [] };
}

/**
 * The editor's document: the graph, who has the newer copy of it, and when to write it back.
 *
 * Three things here are load-bearing, all for the same reason — **workflows are the one object in
 * the app with no history, so a lost update is unrecoverable**:
 *
 * 1. **A 409 is not a generic error.** `PUT /workflows/{id}` returns it when another tab saved
 *    first *or* while the run is `running`/`cancelling`. Either way the local graph is now the
 *    minority opinion, so the save loop stops dead and {@link useWorkflowEditor.conflict} is
 *    raised for the view to offer "Reload". Retrying automatically would be the lost update.
 * 2. **Live refreshes never overwrite unsaved edits.** {@link useWorkflowEditor.applyLive} takes
 *    `state`/`stale` from every poll — that is the whole point of polling — but takes `graph` and
 *    `version` only while the document is clean. A dirty editor keeps its own graph *and its own
 *    stale version*, so the next save 409s honestly instead of silently winning.
 * 3. **Read-only while the runner owns the workflow.** The server refuses the write anyway; doing
 *    it here means the user is told "Stop to edit" instead of losing a keystroke to a 409.
 */
export function useWorkflowEditor(workflowId: Ref<string>) {
  const graph = ref<WorkflowGraph>(emptyGraph());
  const state = ref<WorkflowState>({});
  const stale = ref<Record<string, boolean>>({});
  const version = ref(0);
  /** The row's `name` column, which the list view shows. Not necessarily `graph.name` — see below. */
  const name = ref("");
  const updatedAt = ref("");

  const loading = ref(false);
  const error = ref("");
  const saving = ref(false);
  const dirty = ref(false);
  /** Non-empty when the server refused the last save; the view offers "Reload". */
  const conflict = ref("");

  const readOnly = computed(() => isBusy(state.value.status));

  let saveTimer: ReturnType<typeof setTimeout> | null = null;
  /** Bumped by every edit; a save that lands on a stale revision leaves `dirty` set. */
  let revision = 0;

  function clearTimer(): void {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = null;
  }

  function applyDetail(detail: WorkflowDetail): void {
    graph.value = detail.graph ?? emptyGraph();
    state.value = detail.state ?? {};
    stale.value = detail.stale ?? {};
    version.value = detail.version ?? 0;
    name.value = detail.name ?? "";
    updatedAt.value = detail.updated_at ?? "";
  }

  /**
   * Merge a payload that arrived from the live stream rather than from a user action.
   *
   * `state` and `stale` always land — progress is why we polled. The graph lands only when there
   * is nothing local to lose.
   */
  function applyLive(detail: WorkflowDetail): void {
    state.value = detail.state ?? {};
    stale.value = detail.stale ?? {};
    updatedAt.value = detail.updated_at ?? updatedAt.value;
    if (dirty.value || saving.value) return;
    graph.value = detail.graph ?? graph.value;
    version.value = detail.version ?? version.value;
    name.value = detail.name ?? name.value;
  }

  async function load(): Promise<void> {
    const id = workflowId.value;
    if (!id) return;
    loading.value = true;
    error.value = "";
    try {
      applyDetail(await api.getWorkflow(id));
      dirty.value = false;
      conflict.value = "";
      revision += 1;
    } catch (e) {
      error.value = formatError(e);
    } finally {
      loading.value = false;
    }
  }

  /** Discard local edits and take the server's copy — the "Reload" action on a conflict. */
  async function reload(): Promise<void> {
    clearTimer();
    dirty.value = false;
    await load();
  }

  async function save(): Promise<boolean> {
    const id = workflowId.value;
    if (!id || !dirty.value || saving.value || conflict.value) return false;
    if (readOnly.value) return false;

    clearTimer();
    const attempted = revision;
    saving.value = true;
    try {
      const detail = await api.updateWorkflow(id, {
        graph: graph.value,
        version: version.value,
      });
      version.value = detail.version ?? version.value;
      state.value = detail.state ?? state.value;
      stale.value = detail.stale ?? stale.value;
      name.value = detail.name ?? name.value;
      updatedAt.value = detail.updated_at ?? updatedAt.value;
      // An edit made while the request was in flight keeps the document dirty; the debounce that
      // edit scheduled writes it next.
      if (revision === attempted) dirty.value = false;
      error.value = "";
      return true;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        conflict.value =
          formatError(e) ||
          "Someone else saved this workflow first, or it started running. Reload to continue.";
      } else {
        error.value = formatError(e);
      }
      return false;
    } finally {
      saving.value = false;
    }
  }

  function schedule(): void {
    clearTimer();
    saveTimer = setTimeout(() => {
      saveTimer = null;
      void save();
    }, SAVE_DEBOUNCE_MS);
  }

  /** Write the pending edit now instead of waiting out the debounce (navigation, Run). */
  async function flush(): Promise<boolean> {
    clearTimer();
    if (!dirty.value) return true;
    return save();
  }

  /**
   * Apply a pure graph edit from `lib/workflowGraph` and schedule the write.
   *
   * Refused while read-only: the runner owns the graph then, and a local edit that the server will
   * reject is worse than no edit at all.
   */
  function mutate(fn: (current: WorkflowGraph) => WorkflowGraph): void {
    if (readOnly.value || conflict.value) return;
    graph.value = fn(graph.value);
    revision += 1;
    dirty.value = true;
    schedule();
  }

  function setName(next: string): void {
    mutate((current) => ({ ...current, name: next }));
  }

  function warnOnUnload(event: BeforeUnloadEvent): void {
    if (!dirty.value && !saving.value) return;
    event.preventDefault();
    event.returnValue = "";
  }

  window.addEventListener("beforeunload", warnOnUnload);

  watch(workflowId, () => void load(), { immediate: true });

  onBeforeUnmount(() => {
    window.removeEventListener("beforeunload", warnOnUnload);
    clearTimer();
    // Best effort: a pending edit is written on the way out rather than dropped.
    if (dirty.value) void save();
  });

  return {
    graph,
    state,
    stale,
    version,
    name,
    updatedAt,
    loading,
    error,
    saving,
    dirty,
    conflict,
    readOnly,
    load,
    reload,
    applyLive,
    mutate,
    setName,
    save,
    flush,
  };
}

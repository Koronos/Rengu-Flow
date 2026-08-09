/**
 * The editor document's three unrecoverable-by-design behaviours.
 *
 * A workflow is the only object in the app with no history, so a lost update cannot be undone.
 * That single fact is why the 409 stops the save loop dead, why a live refresh never lands on a
 * dirty document, and why the editor goes read-only while the runner owns the graph. None of the
 * three is visible in a pure module, so this is where they get pinned.
 */
import { createApp, defineComponent, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getWorkflow, updateWorkflow } = vi.hoisted(() => ({
  getWorkflow: vi.fn(),
  updateWorkflow: vi.fn(),
}));

// The real `ApiError` is kept: `useWorkflowEditor` branches on `instanceof ApiError`, so a
// look-alike stub would make the 409 test pass for the wrong reason.
vi.mock("../api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api")>();
  return { ApiError: actual.ApiError, api: { getWorkflow, updateWorkflow } };
});

import { ApiError } from "../api";
import type { WorkflowDetail, WorkflowGraph, WorkflowState } from "../types/workflow";
import { emptyGraph, useWorkflowEditor } from "./useWorkflowEditor";

function graphNamed(name: string): WorkflowGraph {
  return { ...emptyGraph(), name };
}

function makeDetail(over: Partial<WorkflowDetail> = {}): WorkflowDetail {
  return {
    id: 5,
    name: "Re-tag",
    graph: emptyGraph(),
    state: {} as WorkflowState,
    stale: {},
    version: 3,
    created_at: "",
    updated_at: "2026-08-09T10:00:00+00:00",
    ...over,
  };
}

function mountEditor(id = "5") {
  const workflowId = ref(id);
  let editor: ReturnType<typeof useWorkflowEditor> | null = null;
  const Host = defineComponent({
    setup() {
      editor = useWorkflowEditor(workflowId);
      return () => null;
    },
  });
  const app = createApp(Host);
  app.mount(document.createElement("div"));
  return { editor: editor as unknown as ReturnType<typeof useWorkflowEditor>, app, workflowId };
}

/** Drain the microtask queue so the awaited continuations inside load/save run. */
const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

describe("useWorkflowEditor", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    getWorkflow.mockReset();
    updateWorkflow.mockReset();
    getWorkflow.mockResolvedValue(makeDetail());
    updateWorkflow.mockResolvedValue(makeDetail({ version: 4 }));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ---------------------------------------------------------------- the 409

  it("stops the save loop on a 409 and never retries it", async () => {
    updateWorkflow.mockRejectedValue(new ApiError("Someone else saved this workflow first", 409));
    const { editor, app } = mountEditor();
    await flush();

    editor.mutate((current) => ({ ...current, name: "mine" }));
    vi.advanceTimersByTime(700);
    await flush();

    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    expect(editor.conflict.value).toContain("Someone else saved this workflow first");
    // A conflict is not a generic failure: the view offers Reload, not a red toast.
    expect(editor.error.value).toBe("");

    // Neither a further edit, nor any amount of waiting, nor the unmount write-out may produce a
    // second PUT — an automatic retry *is* the lost update.
    editor.mutate((current) => ({ ...current, name: "mine again" }));
    vi.advanceTimersByTime(60_000);
    await flush();
    app.unmount();
    await flush();

    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    expect(editor.graph.value.name).toBe("mine");
  });

  it("treats a non-409 failure as retryable: error, no conflict, next edit saves", async () => {
    updateWorkflow.mockRejectedValueOnce(new ApiError("Server exploded", 500));
    const { editor, app } = mountEditor();
    await flush();

    editor.mutate((current) => ({ ...current, name: "one" }));
    vi.advanceTimersByTime(700);
    await flush();

    expect(editor.error.value).toBe("Server exploded");
    expect(editor.conflict.value).toBe("");
    expect(editor.dirty.value).toBe(true);

    editor.mutate((current) => ({ ...current, name: "two" }));
    vi.advanceTimersByTime(700);
    await flush();

    expect(updateWorkflow).toHaveBeenCalledTimes(2);
    expect(editor.dirty.value).toBe(false);
    app.unmount();
  });

  // ---------------------------------------------------------------- applyLive

  it("applyLive keeps state and stale but never overwrites unsaved edits", async () => {
    const { editor, app } = mountEditor();
    await flush();

    editor.mutate((current) => ({ ...current, name: "local edit" }));

    editor.applyLive(
      makeDetail({
        version: 99,
        name: "server row",
        graph: graphNamed("server graph"),
        state: { status: "idle", current_node: "n2" },
        stale: { n1: true },
      })
    );

    // The graph and its version are the user's; taking the server's would be the lost update, and
    // keeping the stale version is what makes the next save 409 honestly.
    expect(editor.graph.value.name).toBe("local edit");
    expect(editor.version.value).toBe(3);
    // Progress lands regardless — it is the whole reason for the live stream.
    expect(editor.state.value.current_node).toBe("n2");
    expect(editor.stale.value).toEqual({ n1: true });

    // Once the edit is written, the document is clean and the server's copy is welcome again.
    vi.advanceTimersByTime(700);
    await flush();
    expect(editor.dirty.value).toBe(false);

    editor.applyLive(makeDetail({ version: 99, graph: graphNamed("server graph") }));
    expect(editor.graph.value.name).toBe("server graph");
    expect(editor.version.value).toBe(99);
    app.unmount();
  });

  // ---------------------------------------------------------------- the debounce

  it("coalesces a burst of edits into one PUT carrying the version in hand", async () => {
    const { editor, app } = mountEditor();
    await flush();

    editor.mutate((current) => ({ ...current, name: "a" }));
    vi.advanceTimersByTime(300);
    editor.mutate((current) => ({ ...current, name: "ab" }));
    vi.advanceTimersByTime(300);
    editor.mutate((current) => ({ ...current, name: "abc" }));

    // Each edit restarts the wait; typing a folder path is one save, not one save per keystroke.
    vi.advanceTimersByTime(699);
    expect(updateWorkflow).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    await flush();

    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    expect(updateWorkflow).toHaveBeenCalledWith("5", {
      graph: expect.objectContaining({ name: "abc" }),
      version: 3,
    });
    expect(editor.dirty.value).toBe(false);

    // The version the server handed back is what the next save must send; replaying the old one
    // would 409 against our own write.
    editor.mutate((current) => ({ ...current, name: "abcd" }));
    vi.advanceTimersByTime(700);
    await flush();

    expect(updateWorkflow).toHaveBeenCalledTimes(2);
    expect(updateWorkflow).toHaveBeenLastCalledWith("5", {
      graph: expect.objectContaining({ name: "abcd" }),
      version: 4,
    });
    app.unmount();
  });

  it("flush() writes the pending edit now instead of waiting out the debounce", async () => {
    const { editor, app } = mountEditor();
    await flush();

    editor.mutate((current) => ({ ...current, name: "before Run" }));
    const written = editor.flush();
    await flush();

    expect(await written).toBe(true);
    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    expect(editor.dirty.value).toBe(false);

    // The debounce it cancelled must not fire a second, redundant PUT.
    vi.advanceTimersByTime(5_000);
    await flush();
    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    app.unmount();
  });

  // ---------------------------------------------------------------- read-only

  it("refuses every edit while the runner owns the workflow", async () => {
    getWorkflow.mockResolvedValue(makeDetail({ state: { status: "running" } }));
    const { editor, app } = mountEditor();
    await flush();

    expect(editor.readOnly.value).toBe(true);
    editor.mutate((current) => ({ ...current, name: "sneaky" }));
    editor.setName("sneakier");

    expect(editor.graph.value.name).toBe("");
    expect(editor.dirty.value).toBe(false);
    vi.advanceTimersByTime(5_000);
    await flush();
    expect(updateWorkflow).not.toHaveBeenCalled();

    // `cancelling` owns the graph just as much as `running` does.
    editor.applyLive(makeDetail({ state: { status: "cancelling" } }));
    expect(editor.readOnly.value).toBe(true);
    editor.mutate((current) => ({ ...current, name: "still sneaky" }));
    expect(editor.graph.value.name).toBe("");

    // ...and the moment the run lets go, editing works again.
    editor.applyLive(makeDetail({ state: { status: "done" } }));
    expect(editor.readOnly.value).toBe(false);
    editor.mutate((current) => ({ ...current, name: "allowed" }));
    expect(editor.graph.value.name).toBe("allowed");
    vi.advanceTimersByTime(700);
    await flush();
    expect(updateWorkflow).toHaveBeenCalledTimes(1);
    app.unmount();
  });
});

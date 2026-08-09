/**
 * The live half of the workflow editor: the socket is a *change signal*, and the reconnect is the
 * only thing standing between a dropped connection and a run bar frozen on yesterday's state.
 *
 * Nothing here is derivable from a pure module — every assertion is about state that outlives a
 * single call: the socket generation, the backoff delay, the progress reading that must not carry
 * over between nodes, and the pre-flight list that a multi-line failure fills.
 */
import { createApp, defineComponent, nextTick, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { getWorkflow, startWorkflow, cancelWorkflow, validateWorkflow, workflowNodeLog } =
  vi.hoisted(() => ({
    getWorkflow: vi.fn(),
    startWorkflow: vi.fn(),
    cancelWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    workflowNodeLog: vi.fn(),
  }));
const { messageError, messageSuccess } = vi.hoisted(() => ({
  messageError: vi.fn(),
  messageSuccess: vi.fn(),
}));

vi.mock("../api", () => ({
  api: { getWorkflow, startWorkflow, cancelWorkflow, validateWorkflow, workflowNodeLog },
}));
vi.mock("element-plus", () => ({
  ElMessage: { error: messageError, success: messageSuccess },
}));

import type { WorkflowDetail } from "../types/workflow";
import { useWorkflowRun } from "./useWorkflowRun";

/** A socket that only does what the test tells it to; every instance is kept for inspection. */
const sockets: FakeWebSocket[] = [];
class FakeWebSocket {
  onopen: ((ev?: unknown) => void) | null = null;
  onmessage: ((ev?: unknown) => void) | null = null;
  onerror: ((ev?: unknown) => void) | null = null;
  onclose: ((ev?: unknown) => void) | null = null;
  closed = false;
  readonly url: string;
  constructor(url: string) {
    this.url = url;
    sockets.push(this);
  }
  close(): void {
    this.closed = true;
  }
}

function makeDetail(over: Partial<WorkflowDetail> = {}): WorkflowDetail {
  return {
    id: 5,
    name: "Re-tag",
    graph: { version: 1, name: "", description: "", variables: [], nodes: [] },
    state: {},
    stale: {},
    version: 3,
    created_at: "",
    updated_at: "",
    ...over,
  };
}

function mountRun(options?: { running?: boolean; currentNode?: string | null }) {
  const workflowId = ref("5");
  const running = ref(options?.running ?? false);
  const currentNode = ref<string | null>(options?.currentNode ?? null);
  const details: WorkflowDetail[] = [];
  let run: ReturnType<typeof useWorkflowRun> | null = null;
  const Host = defineComponent({
    setup() {
      run = useWorkflowRun({
        workflowId,
        running,
        currentNode,
        onDetail: (detail) => details.push(detail),
      });
      return () => null;
    },
  });
  const app = createApp(Host);
  app.mount(document.createElement("div"));
  return {
    run: run as unknown as ReturnType<typeof useWorkflowRun>,
    app,
    details,
    workflowId,
    running,
    currentNode,
  };
}

const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const latest = () => sockets[sockets.length - 1];

describe("useWorkflowRun", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sockets.length = 0;
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
    getWorkflow.mockReset().mockResolvedValue(makeDetail());
    startWorkflow.mockReset().mockResolvedValue(makeDetail());
    cancelWorkflow.mockReset().mockResolvedValue(makeDetail());
    validateWorkflow.mockReset().mockResolvedValue({ errors: [] });
    workflowNodeLog.mockReset().mockResolvedValue({ chunk: "", offset: 0, progress: null });
    messageError.mockReset();
    messageSuccess.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ---------------------------------------------------------------- the change signal

  it("answers a workflows-changed frame with one GET and applies the payload", async () => {
    const { run, app, details } = mountRun();

    expect(sockets).toHaveLength(1);
    expect(latest().url).toContain("/api/v1/workflows/events/ws");
    expect(run.streamStatus.value).toBe("offline");

    latest().onopen?.();
    expect(run.streamStatus.value).toBe("connected");

    latest().onmessage?.({ data: JSON.stringify({ type: "workflows-changed", version: 12 }) });
    await flush();

    expect(getWorkflow).toHaveBeenCalledTimes(1);
    expect(details).toHaveLength(1);
    expect(details[0].version).toBe(3);

    // Frames the socket may carry that are not the change signal, and outright garbage, must be
    // ignored rather than turned into a refresh storm or an exception in the handler.
    latest().onmessage?.({ data: JSON.stringify({ type: "something-else" }) });
    latest().onmessage?.({ data: "not json at all" });
    await flush();
    expect(getWorkflow).toHaveBeenCalledTimes(1);

    // A refresh that fails is swallowed: the next frame retries and a toast per blip is noise.
    getWorkflow.mockRejectedValueOnce(new Error("offline"));
    latest().onmessage?.({ data: JSON.stringify({ type: "workflows-changed" }) });
    await flush();
    expect(details).toHaveLength(1);

    app.unmount();
  });

  // ---------------------------------------------------------------- reconnection

  it("reconnects with exponential backoff and resets the delay once a socket opens", async () => {
    const { run, app } = mountRun();
    expect(sockets).toHaveLength(1);

    // First drop: reconnect after the 1 s floor, not immediately.
    latest().onclose?.();
    expect(run.streamStatus.value).toBe("reconnecting");
    vi.advanceTimersByTime(999);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);

    // Second drop without ever connecting: the wait doubles.
    latest().onclose?.();
    vi.advanceTimersByTime(1999);
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(3);

    // A socket that actually opens puts the delay back on the floor, so a long outage followed by
    // a brief blip does not leave the user staring at a 15 s gap.
    latest().onopen?.();
    expect(run.streamStatus.value).toBe("connected");
    latest().onclose?.();
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(4);

    app.unmount();
  });

  it("stops reconnecting once the view is gone", async () => {
    const { app } = mountRun();
    latest().onclose?.();
    app.unmount();

    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(1);
  });

  it("detaches the old socket when the workflow changes, so it cannot reconnect", async () => {
    const { app, workflowId } = mountRun();
    const first = latest();

    workflowId.value = "6";
    await nextTick();
    expect(sockets).toHaveLength(2);
    expect(first.closed).toBe(true);

    // Detached, not merely closed: the browser still delivers a close event after `close()`, and a
    // live handler on the superseded socket would schedule a reconnect for the workflow we left.
    expect(first.onclose).toBeNull();
    expect(first.onmessage).toBeNull();

    vi.advanceTimersByTime(60_000);
    expect(sockets).toHaveLength(2);

    app.unmount();
  });

  // ---------------------------------------------------------------- progress

  it("polls the running node's log and starts each node from zero", async () => {
    workflowNodeLog.mockResolvedValue({ chunk: "", offset: 120, progress: { percent: 40 } });
    const { run, app, currentNode } = mountRun({ running: true, currentNode: "n1" });
    await flush();

    expect(workflowNodeLog).toHaveBeenCalledWith("5", "n1", 0);
    expect(run.progress.value).toEqual({ percent: 40 });

    // The poll continues from the offset the server handed back.
    vi.advanceTimersByTime(2000);
    await flush();
    expect(workflowNodeLog).toHaveBeenLastCalledWith("5", "n1", 120);

    // A new node starts blank: carrying the previous node's 40 % over would read as progress on a
    // step that has not begun, and its offset would skip the head of the new log. The next poll is
    // left hanging so the assertion is about the reset, not about what the poll answered.
    workflowNodeLog.mockReturnValue(new Promise(() => {}));
    currentNode.value = "n2";
    await nextTick();
    await flush();
    expect(run.progress.value).toBeNull();
    expect(workflowNodeLog).toHaveBeenLastCalledWith("5", "n2", 0);

    app.unmount();
  });

  it("stops polling when the run stops", async () => {
    const { app, running } = mountRun({ running: true, currentNode: "n1" });
    await flush();
    const callsWhileRunning = workflowNodeLog.mock.calls.length;

    running.value = false;
    await nextTick();
    vi.advanceTimersByTime(10_000);
    await flush();

    expect(workflowNodeLog.mock.calls.length).toBe(callsWhileRunning);
    app.unmount();
  });

  // ---------------------------------------------------------------- pre-flight reporting

  it("puts a multi-line failure in the pre-flight banner and a single line in a toast", async () => {
    const { run, app } = mountRun();

    startWorkflow.mockRejectedValueOnce(
      new Error("node n2 · index stage needs a model\nnode n3 · has no source")
    );
    expect(await run.start()).toBe(false);
    expect(run.preflight.value).toEqual([
      "node n2 · index stage needs a model",
      "node n3 · has no source",
    ]);
    expect(messageError).toHaveBeenCalledWith("2 problems stop this workflow from running");

    // A single-line failure is a toast, and it clears the banner rather than leaving a stale list.
    startWorkflow.mockRejectedValueOnce(new Error("This workflow is already running."));
    expect(await run.start()).toBe(false);
    expect(run.preflight.value).toEqual([]);
    expect(messageError).toHaveBeenLastCalledWith("This workflow is already running.");

    app.unmount();
  });

  it("keeps the pre-flight list from an explicit validate, and clears it on a successful start", async () => {
    const { run, app, details } = mountRun();

    validateWorkflow.mockResolvedValueOnce({ errors: ["node n2 · index stage needs a model"] });
    expect(await run.validate()).toEqual(["node n2 · index stage needs a model"]);
    expect(run.preflight.value).toHaveLength(1);
    expect(messageSuccess).not.toHaveBeenCalled();

    expect(await run.start()).toBe(true);
    expect(run.preflight.value).toEqual([]);
    expect(details).toHaveLength(1);

    await run.validate();
    expect(messageSuccess).toHaveBeenCalledWith("Pre-flight found no problems");
    app.unmount();
  });
});

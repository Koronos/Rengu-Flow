import { onUnmounted, ref, watch, type Ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { wsBaseUrl } from "../lib/wsLog";
import type { RunProgress } from "../types/api";
import type { WorkflowDetail, WorkflowStartOptions } from "../types/workflow";

const RECONNECT_MS_MIN = 1000;
const RECONNECT_MS_MAX = 15000;
/** The node log is the only source of a percentage; the state blob carries statuses, not progress. */
const PROGRESS_POLL_MS = 2000;

export type WorkflowStreamStatus = "connected" | "reconnecting" | "offline";

export interface UseWorkflowRunOptions {
  workflowId: Ref<string>;
  /** Whether a node is currently executing — gates the progress poll. */
  running: Ref<boolean>;
  /** `state.current_node`; changing it resets the progress reading. */
  currentNode: Ref<string | null>;
  /** Called with every fresh payload: from the event stream and from every action. */
  onDetail: (detail: WorkflowDetail) => void;
}

/**
 * Live workflow state plus the five run actions.
 *
 * The socket (`/workflows/events/ws`) is a **change signal, not a state feed** — the server sends
 * `{type: "workflows-changed", version}` whenever `workflows_version()` moves, which
 * `workflow_db.mutate_state` bumps on every write the runner makes. So each frame is answered with
 * one `GET /workflows/{id}`, and the payload that comes back carries `stale` computed **on the
 * server**. That is deliberate and not a round-trip we can save: `JSON.stringify` emits `80` where
 * Python emits `80.0`, so a client-side hash would disagree with the server on every
 * `prep.quality` node. Reconnection uses the same exponential backoff as `useJobsEvents`.
 */
export function useWorkflowRun(options: UseWorkflowRunOptions) {
  const { workflowId, running, currentNode, onDetail } = options;

  const streamStatus = ref<WorkflowStreamStatus>("offline");
  const progress = ref<RunProgress | null>(null);
  const busy = ref(false);
  /** Pre-flight output: every problem at once, never one at a time. */
  const preflight = ref<string[]>([]);

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = RECONNECT_MS_MIN;
  let gen = 0;

  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let logOffset = 0;

  // ------------------------------------------------------------------ refresh

  async function refresh(): Promise<void> {
    const id = workflowId.value;
    if (!id) return;
    try {
      onDetail(await api.getWorkflow(id));
    } catch {
      // A dropped refresh is not worth a toast: the next socket frame or poll retries.
    }
  }

  // ------------------------------------------------------------------ progress

  async function pollProgress(): Promise<void> {
    const id = workflowId.value;
    const node = currentNode.value;
    if (!id || !node) return;
    try {
      const result = await api.workflowNodeLog(id, node, logOffset);
      logOffset = result.offset ?? logOffset;
      if (result.progress) progress.value = result.progress;
    } catch {
      /* the node directory may not exist yet; the next tick retries */
    }
  }

  function stopPolling(): void {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function startPolling(): void {
    stopPolling();
    void pollProgress();
    pollTimer = setInterval(() => void pollProgress(), PROGRESS_POLL_MS);
  }

  watch(currentNode, () => {
    // A new node starts from zero; carrying the previous node's 100 % over would read as instant
    // completion of a step that has not begun.
    logOffset = 0;
    progress.value = null;
  });

  watch(
    [running, currentNode],
    ([isRunning, node]) => {
      if (isRunning && node) startPolling();
      else stopPolling();
    },
    { immediate: true }
  );

  // ------------------------------------------------------------------ socket

  function clearReconnect(): void {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function disconnect(): void {
    gen += 1;
    clearReconnect();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
      ws = null;
    }
    streamStatus.value = "offline";
  }

  function scheduleReconnect(myGen: number): void {
    if (myGen !== gen) return;
    streamStatus.value = "reconnecting";
    clearReconnect();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (myGen !== gen) return;
      connect();
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MS_MAX);
    }, reconnectDelay);
  }

  function connect(): void {
    disconnect();
    const myGen = gen;
    try {
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/workflows/events/ws`);
    } catch {
      scheduleReconnect(myGen);
      return;
    }
    ws.onopen = () => {
      if (myGen !== gen) return;
      streamStatus.value = "connected";
      reconnectDelay = RECONNECT_MS_MIN;
    };
    ws.onmessage = (event) => {
      if (myGen !== gen) return;
      try {
        const data = JSON.parse(String(event.data ?? "")) as { type?: string };
        if (data?.type === "workflows-changed") void refresh();
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onclose = () => {
      if (myGen !== gen) return;
      ws = null;
      scheduleReconnect(myGen);
    };
  }

  // ------------------------------------------------------------------ actions

  /**
   * Report a failed action.
   *
   * A multi-line message is pre-flight's "everything wrong at once" list, which belongs in the
   * banner where it can be read and worked through; a single line is a toast.
   */
  function reportFailure(e: unknown): void {
    const message = formatError(e);
    const lines = message.split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.length > 1) {
      preflight.value = lines;
      ElMessage.error(`${lines.length} problems stop this workflow from running`);
    } else {
      preflight.value = [];
      ElMessage.error(message);
    }
  }

  async function start(startOptions?: WorkflowStartOptions): Promise<boolean> {
    const id = workflowId.value;
    if (!id || busy.value) return false;
    busy.value = true;
    try {
      onDetail(await api.startWorkflow(id, startOptions));
      preflight.value = [];
      return true;
    } catch (e) {
      reportFailure(e);
      return false;
    } finally {
      busy.value = false;
    }
  }

  async function stop(): Promise<boolean> {
    const id = workflowId.value;
    if (!id || busy.value) return false;
    busy.value = true;
    try {
      onDetail(await api.cancelWorkflow(id));
      return true;
    } catch (e) {
      reportFailure(e);
      return false;
    } finally {
      busy.value = false;
    }
  }

  async function validate(): Promise<string[]> {
    const id = workflowId.value;
    if (!id || busy.value) return [];
    busy.value = true;
    try {
      const result = await api.validateWorkflow(id);
      preflight.value = result.errors ?? [];
      if (!preflight.value.length) ElMessage.success("Pre-flight found no problems");
      return preflight.value;
    } catch (e) {
      reportFailure(e);
      return preflight.value;
    } finally {
      busy.value = false;
    }
  }

  watch(
    workflowId,
    (id) => {
      progress.value = null;
      logOffset = 0;
      preflight.value = [];
      if (id) connect();
      else disconnect();
    },
    { immediate: true }
  );

  onUnmounted(() => {
    disconnect();
    stopPolling();
  });

  return {
    streamStatus,
    progress,
    busy,
    preflight,
    refresh,
    start,
    stop,
    validate,
    reconnect: connect,
  };
}

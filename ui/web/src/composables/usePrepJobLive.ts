import { computed, onUnmounted, ref, watch, type Ref } from "vue";
import { appendBoundedLogChunk, wsBaseUrl } from "../lib/wsLog";

const RECONNECT_MS_MIN = 1000;
const RECONNECT_MS_MAX = 15000;

export type PrepLiveStreamStatus = "connected" | "reconnecting" | "offline";

/** Progress payload for a prep job's live stream (phase: "prep:tag" | "prep:caption" | "prep:clean"). */
export interface PrepProgress {
  phase?: string | null;
  step?: number | null;
  max_steps?: number | null;
  msg?: string | null;
  percent?: number | null;
}

type LiveMessage =
  | { type: "progress"; state?: string; progress: PrepProgress | null }
  | { type: "log_line"; chunk: string }
  | { type: "run_finished"; state: string }
  | { type: "error"; message: string };

function parseMessage(raw: string): LiveMessage | null {
  try {
    const data = JSON.parse(raw) as LiveMessage;
    if (data && typeof data === "object" && "type" in data) return data;
  } catch {
    /* ignore */
  }
  return null;
}

/** Lightweight live-stream composable for prep jobs — progress + log tail via WebSocket. */
export function usePrepJobLive(
  jobId: Ref<string | undefined> | (() => string | undefined),
  options?: { onRunFinished?: () => void }
) {
  const progress = ref<PrepProgress | null>(null);
  const logChunks = ref<string[]>([]);
  const streamStatus = ref<PrepLiveStreamStatus>("offline");
  const streamError = ref("");

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = RECONNECT_MS_MIN;
  let connectionGen = 0;
  const charCount = { value: 0 };

  const logText = computed(() => logChunks.value.join(""));

  function resolveJobId(): string {
    const id = typeof jobId === "function" ? jobId() : jobId.value;
    return id?.trim() || "";
  }

  function appendLog(text: string): void {
    appendBoundedLogChunk(logChunks.value, charCount, text);
  }

  function clearReconnectTimer(): void {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function disconnect(): void {
    connectionGen += 1;
    clearReconnectTimer();
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

  function reset(): void {
    disconnect();
    progress.value = null;
    logChunks.value = [];
    charCount.value = 0;
    streamError.value = "";
    reconnectDelay = RECONNECT_MS_MIN;
  }

  function scheduleReconnect(gen: number): void {
    if (gen !== connectionGen || !resolveJobId()) return;
    streamStatus.value = "reconnecting";
    clearReconnectTimer();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (gen !== connectionGen) return;
      connect();
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MS_MAX);
    }, reconnectDelay);
  }

  function handleMessage(msg: LiveMessage): void {
    switch (msg.type) {
      case "progress":
        progress.value = msg.progress ?? null;
        break;
      case "log_line":
        appendLog(msg.chunk);
        break;
      case "run_finished":
        options?.onRunFinished?.();
        disconnect();
        break;
      case "error":
        streamError.value = msg.message;
        disconnect();
        break;
      default:
        break;
    }
  }

  function connect(): void {
    const id = resolveJobId();
    if (!id) return;
    disconnect();
    const gen = connectionGen;
    streamError.value = "";
    try {
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/jobs/${encodeURIComponent(id)}/live/ws`);
    } catch (e) {
      streamError.value = e instanceof Error ? e.message : "Could not open live stream";
      scheduleReconnect(gen);
      return;
    }
    ws.onopen = () => {
      if (gen !== connectionGen) return;
      streamStatus.value = "connected";
      reconnectDelay = RECONNECT_MS_MIN;
    };
    ws.onmessage = (event) => {
      if (gen !== connectionGen) return;
      const msg = parseMessage(String(event.data ?? ""));
      if (msg) handleMessage(msg);
    };
    ws.onerror = () => {
      if (gen !== connectionGen) return;
      streamError.value = "Live stream connection error";
    };
    ws.onclose = () => {
      if (gen !== connectionGen) return;
      ws = null;
      scheduleReconnect(gen);
    };
  }

  watch(
    () => resolveJobId(),
    (id, prev) => {
      if (id === prev) return;
      reset();
      if (id) connect();
    },
    { immediate: true }
  );

  onUnmounted(reset);

  return {
    progress,
    logText,
    streamStatus,
    streamError,
    reconnect: connect,
    disconnect,
  };
}

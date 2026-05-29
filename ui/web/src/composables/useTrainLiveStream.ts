import { computed, onUnmounted, ref, watch, type Ref } from "vue";
import { api } from "../api";
import { appendBoundedLogChunk, wsBaseUrl } from "../lib/wsLog";
import type { RunPreviewImageRef, RunProgress } from "../types/api";

const RECONNECT_MS_MIN = 1000;
const RECONNECT_MS_MAX = 15000;
const LOG_POLL_MS = 2000;

export type LiveStreamStatus = "connected" | "reconnecting" | "offline";

export type JobLiveMessage =
  | { type: "progress"; state: string; run_dir?: string | null; progress: RunProgress | null }
  | { type: "log_line"; chunk: string }
  | {
      type: "metrics";
      scalars: Record<string, { step: number; value: number }[]>;
      preview_images: RunPreviewImageRef[];
    }
  | { type: "run_finished"; state: string }
  | { type: "error"; message: string };

function parseMessage(raw: string): JobLiveMessage | null {
  try {
    const data = JSON.parse(raw) as JobLiveMessage;
    if (data && typeof data === "object" && "type" in data) return data;
  } catch {
    /* ignore */
  }
  return null;
}

/** Live training stream: progress, metrics, and log tail over WebSocket with HTTP fallback. */
export function useTrainLiveStream(
  jobId: Ref<string | undefined> | (() => string | undefined),
  options?: { onRunFinished?: () => void }
) {
  const progress = ref<RunProgress | null>(null);
  const scalars = ref<Record<string, { step: number; value: number }[]>>({});
  const previewImages = ref<RunPreviewImageRef[]>([]);
  const logChunks = ref<string[]>([]);
  const streamStatus = ref<LiveStreamStatus>("offline");
  const useHttpFallback = ref(true);
  const streamError = ref("");

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let logPollTimer: ReturnType<typeof setInterval> | null = null;
  let logOffset = 0;
  const charCount = { value: 0 };
  let reconnectDelay = RECONNECT_MS_MIN;
  let connectionGen = 0;

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

  function stopLogPoll(): void {
    if (logPollTimer) clearInterval(logPollTimer);
    logPollTimer = null;
  }

  function disconnect(): void {
    connectionGen += 1;
    clearReconnectTimer();
    stopLogPoll();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
      ws = null;
    }
    streamStatus.value = "offline";
    useHttpFallback.value = true;
  }

  function reset(): void {
    disconnect();
    progress.value = null;
    scalars.value = {};
    previewImages.value = [];
    logChunks.value = [];
    charCount.value = 0;
    logOffset = 0;
    streamError.value = "";
    reconnectDelay = RECONNECT_MS_MIN;
  }

  function scheduleReconnect(gen: number): void {
    if (gen !== connectionGen || !resolveJobId()) return;
    streamStatus.value = "reconnecting";
    useHttpFallback.value = true;
    clearReconnectTimer();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (gen !== connectionGen) return;
      connect();
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MS_MAX);
    }, reconnectDelay);
  }

  async function pollLogsHttp(): Promise<void> {
    const id = resolveJobId();
    if (!id || !useHttpFallback.value) return;
    const gen = connectionGen;
    try {
      const data = await api.jobLogs(id, logOffset);
      if (gen !== connectionGen) return; // job switched mid-fetch: drop stale logs/offset
      if (data.chunk) {
        appendLog(data.chunk);
        logOffset = data.offset;
      }
    } catch (e) {
      if (gen !== connectionGen) return;
      streamError.value = e instanceof Error ? e.message : "Log poll failed";
    }
  }

  function startLogPoll(): void {
    stopLogPoll();
    if (!useHttpFallback.value) return;
    void pollLogsHttp();
    logPollTimer = setInterval(() => void pollLogsHttp(), LOG_POLL_MS);
  }

  function handleMessage(msg: JobLiveMessage): void {
    switch (msg.type) {
      case "progress":
        progress.value = msg.progress ?? null;
        break;
      case "log_line":
        appendLog(msg.chunk);
        break;
      case "metrics":
        scalars.value = msg.scalars || {};
        previewImages.value = msg.preview_images || [];
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
      useHttpFallback.value = true;
      startLogPoll();
      scheduleReconnect(gen);
      return;
    }
    ws.onopen = () => {
      if (gen !== connectionGen) return;
      streamStatus.value = "connected";
      useHttpFallback.value = false;
      reconnectDelay = RECONNECT_MS_MIN;
      stopLogPoll();
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
      useHttpFallback.value = true;
      startLogPoll();
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
    scalars,
    previewImages,
    logText,
    streamStatus,
    useHttpFallback,
    streamError,
    reconnect: connect,
    disconnect,
  };
}

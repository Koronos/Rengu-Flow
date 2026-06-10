import { computed, onUnmounted, ref, watch, type Ref } from "vue";
import { api } from "../api";
import { appendBoundedLogChunk, MAX_LOG_CHARS, wsBaseUrl } from "../lib/wsLog";

/**
 * Stream a job's stdout. Primary path is the logs WebSocket; an HTTP poll fallback keeps the log
 * visible when the socket can't connect — e.g. a dev proxy that doesn't forward WS upgrades, a
 * terminal job opened after it finished, or a WS-hostile network. Without the fallback a failed
 * run just shows "(waiting for output…)" and the error is invisible.
 */
export function useJobLogStream(jobId: Ref<string | undefined> | (() => string | undefined)) {
  const chunks = ref<string[]>([]);
  const connected = ref(false);
  const streamError = ref("");
  let ws: WebSocket | null = null;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let httpOffset = 0;
  let wsPrimed = false;
  // Bumped on every teardown (job switch / unmount) so a late HTTP response from a previous
  // job can't append to the new job's log or clobber its offset.
  let fetchGen = 0;
  const charCount = { value: 0 };

  const logText = computed(() => chunks.value.join(""));

  function resolveJobId(): string {
    const id = typeof jobId === "function" ? jobId() : jobId.value;
    return id?.trim() || "";
  }

  function stopPolling(): void {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  // Fetch new log bytes over HTTP, accumulating from httpOffset. Re-checks `connected` after the
  // await so a socket that opens mid-request doesn't get its content duplicated by a late append.
  async function pollHttp(): Promise<void> {
    const id = resolveJobId();
    if (!id || connected.value) return;
    const gen = fetchGen;
    try {
      const { chunk, offset } = await api.jobLogs(id, httpOffset);
      if (gen !== fetchGen) return; // job switched during the await — discard this result
      if (chunk && !connected.value) {
        appendBoundedLogChunk(chunks.value, charCount, chunk);
      }
      httpOffset = offset;
    } catch (e) {
      if (gen !== fetchGen) return;
      streamError.value = e instanceof Error ? e.message : "Could not load log";
    }
    if (gen === fetchGen && !connected.value) {
      pollTimer = setTimeout(pollHttp, 2000);
    }
  }

  function disconnect(): void {
    fetchGen++;
    stopPolling();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
      ws = null;
    }
    connected.value = false;
  }

  function reset(): void {
    disconnect();
    chunks.value = [];
    charCount.value = 0;
    httpOffset = 0;
    wsPrimed = false;
    streamError.value = "";
  }

  function connect(): void {
    const id = resolveJobId();
    if (!id) return;
    disconnect();
    streamError.value = "";
    wsPrimed = false;
    httpOffset = 0;
    // Prime immediately over HTTP so existing output (and errors on terminal runs) shows without
    // waiting on the socket. If the WS connects, it takes over and replaces this content.
    void pollHttp();
    try {
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/jobs/${encodeURIComponent(id)}/logs/ws`);
    } catch (e) {
      streamError.value = e instanceof Error ? e.message : "Could not open log stream";
      return; // HTTP polling continues as the fallback.
    }
    ws.onopen = () => {
      connected.value = true;
      stopPolling();
    };
    ws.onmessage = (event) => {
      // The socket replays from offset 0, so reset once on its first message to avoid duplicating
      // whatever the HTTP prime already appended.
      if (!wsPrimed) {
        chunks.value = [];
        charCount.value = 0;
        wsPrimed = true;
      }
      appendBoundedLogChunk(chunks.value, charCount, String(event.data ?? ""));
    };
    ws.onerror = () => {
      streamError.value = "Log stream connection error";
    };
    ws.onclose = () => {
      connected.value = false;
      // Socket dropped — resume HTTP polling (the run may still be active).
      if (!pollTimer) pollTimer = setTimeout(pollHttp, 2000);
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

  onUnmounted(disconnect);

  return { logText, connected, streamError, reset, reconnect: connect, disconnect };
}

// Re-export for tests that assert the cap constant.
export { MAX_LOG_CHARS };

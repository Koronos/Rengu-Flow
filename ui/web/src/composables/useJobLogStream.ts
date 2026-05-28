import { computed, onUnmounted, ref, watch, type Ref } from "vue";
import { appendBoundedLogChunk, MAX_LOG_CHARS, wsBaseUrl } from "../lib/wsLog";

/** Stream job stdout over the existing logs WebSocket; falls back to empty when disconnected. */
export function useJobLogStream(jobId: Ref<string | undefined> | (() => string | undefined)) {
  const chunks = ref<string[]>([]);
  const connected = ref(false);
  const streamError = ref("");
  let ws: WebSocket | null = null;
  const charCount = { value: 0 };

  const logText = computed(() => chunks.value.join(""));

  function resolveJobId(): string {
    const id = typeof jobId === "function" ? jobId() : jobId.value;
    return id?.trim() || "";
  }

  function disconnect(): void {
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.close();
      ws = null;
    }
    connected.value = false;
  }

  function reset(): void {
    disconnect();
    chunks.value = [];
    charCount.value = 0;
    streamError.value = "";
  }

  function connect(): void {
    const id = resolveJobId();
    if (!id) return;
    disconnect();
    streamError.value = "";
    try {
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/jobs/${encodeURIComponent(id)}/logs/ws`);
    } catch (e) {
      streamError.value = e instanceof Error ? e.message : "Could not open log stream";
      return;
    }
    ws.onopen = () => {
      connected.value = true;
    };
    ws.onmessage = (event) => {
      appendBoundedLogChunk(chunks.value, charCount, String(event.data ?? ""));
    };
    ws.onerror = () => {
      streamError.value = "Log stream connection error";
    };
    ws.onclose = () => {
      connected.value = false;
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

import { onMounted, onUnmounted, ref } from "vue";
import { api } from "../api";
import { wsBaseUrl } from "../lib/wsLog";
import type { SystemStatsResponse } from "../types/api";

const RECONNECT_MS_MIN = 1000;
const RECONNECT_MS_MAX = 15000;
const HTTP_POLL_MS = 2000;

export type SystemStatsStreamStatus = "connected" | "reconnecting" | "offline";

type SystemStatsMessage = { type: "system_stats"; stats: SystemStatsResponse };

function parseMessage(raw: string): SystemStatsMessage | null {
  try {
    const data = JSON.parse(raw) as SystemStatsMessage;
    if (data && typeof data === "object" && data.type === "system_stats") return data;
  } catch {
    /* ignore */
  }
  return null;
}

/**
 * Global host stats (CPU/RAM/GPU) over a single WebSocket, with HTTP polling as a fallback.
 * One socket replaces the per-client 2s polling of GET /system/stats; on disconnect it falls
 * back to polling and reconnects with exponential backoff.
 */
export function useSystemStatsStream() {
  const stats = ref<SystemStatsResponse | null>(null);
  const loading = ref(true);
  const status = ref<SystemStatsStreamStatus>("offline");

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let httpTimer: ReturnType<typeof setInterval> | null = null;
  let reconnectDelay = RECONNECT_MS_MIN;
  let useHttpFallback = true;
  let connectionGen = 0;

  function clearReconnectTimer(): void {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function stopHttpPoll(): void {
    if (httpTimer) clearInterval(httpTimer);
    httpTimer = null;
  }

  async function pollHttp(): Promise<void> {
    if (!useHttpFallback) return;
    const gen = connectionGen;
    try {
      const data = await api.getSystemStats();
      if (gen !== connectionGen || !useHttpFallback) return;
      stats.value = data;
    } catch {
      /* keep last-known stats; the bar shows them until the next success */
    } finally {
      loading.value = false;
    }
  }

  function startHttpPoll(): void {
    stopHttpPoll();
    if (!useHttpFallback) return;
    void pollHttp();
    httpTimer = setInterval(() => void pollHttp(), HTTP_POLL_MS);
  }

  function disconnect(): void {
    connectionGen += 1;
    clearReconnectTimer();
    stopHttpPoll();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
      ws = null;
    }
    status.value = "offline";
    useHttpFallback = true;
  }

  function scheduleReconnect(gen: number): void {
    if (gen !== connectionGen) return;
    status.value = "reconnecting";
    useHttpFallback = true;
    startHttpPoll();
    clearReconnectTimer();
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      if (gen !== connectionGen) return;
      connect();
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MS_MAX);
    }, reconnectDelay);
  }

  function connect(): void {
    disconnect();
    const gen = connectionGen;
    try {
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/system/stats/ws`);
    } catch {
      useHttpFallback = true;
      startHttpPoll();
      scheduleReconnect(gen);
      return;
    }
    ws.onopen = () => {
      if (gen !== connectionGen) return;
      status.value = "connected";
      useHttpFallback = false;
      reconnectDelay = RECONNECT_MS_MIN;
      stopHttpPoll();
    };
    ws.onmessage = (event) => {
      if (gen !== connectionGen) return;
      const msg = parseMessage(String(event.data ?? ""));
      if (msg) {
        stats.value = msg.stats;
        loading.value = false;
      }
    };
    ws.onerror = () => {
      /* surfaced via onclose -> fallback + reconnect */
    };
    ws.onclose = () => {
      if (gen !== connectionGen) return;
      ws = null;
      useHttpFallback = true;
      startHttpPoll();
      scheduleReconnect(gen);
    };
  }

  onMounted(connect);
  onUnmounted(disconnect);

  return { stats, loading, status };
}

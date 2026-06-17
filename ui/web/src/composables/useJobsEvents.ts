import { onUnmounted } from "vue";
import { wsBaseUrl } from "../lib/wsLog";

const RECONNECT_MS_MIN = 1000;
const RECONNECT_MS_MAX = 15000;

/**
 * Subscribe to server-pushed job-list change events (`/api/v1/jobs/events/ws`). `onChange` fires
 * whenever any job row changes — created/updated/deleted, including a run the server's queue poller
 * transitions to finished/failed — so a list view can refresh on demand instead of polling
 * `GET /jobs` on a timer. The server sends one frame on connect, so a reconnect that missed changes
 * reconciles immediately. Reconnects with exponential backoff; cleans up on unmount.
 */
export function useJobsEvents(onChange: () => void) {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = RECONNECT_MS_MIN;
  // Bumped on every (re)connect/disconnect so stale socket callbacks bail out instead of racing.
  let gen = 0;

  function clearTimer(): void {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function disconnect(): void {
    gen += 1;
    clearTimer();
    if (ws) {
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
      ws = null;
    }
  }

  function scheduleReconnect(myGen: number): void {
    if (myGen !== gen) return;
    clearTimer();
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
      ws = new WebSocket(`${wsBaseUrl()}/api/v1/jobs/events/ws`);
    } catch {
      scheduleReconnect(myGen);
      return;
    }
    ws.onopen = () => {
      if (myGen === gen) reconnectDelay = RECONNECT_MS_MIN;
    };
    ws.onmessage = (event) => {
      if (myGen !== gen) return;
      try {
        const data = JSON.parse(String(event.data ?? "")) as { type?: string };
        if (data?.type === "jobs-changed") onChange();
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

  connect();
  onUnmounted(disconnect);

  return { reconnect: connect, disconnect };
}

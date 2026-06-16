import { createApp, defineComponent, nextTick, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { jobLogs } = vi.hoisted(() => ({ jobLogs: vi.fn() }));
vi.mock("../api", () => ({ api: { jobLogs } }));

import { useJobLogStream } from "./useJobLogStream";

// A WebSocket that never opens on its own, so the composable stays on its HTTP-poll fallback
// path — which is exactly where the stale-response race lives. Tests that need to drive the
// socket lifecycle grab the latest instance from `wsInstances` and invoke its handlers.
const wsInstances: FakeWebSocket[] = [];
class FakeWebSocket {
  onopen: ((ev?: unknown) => void) | null = null;
  onmessage: ((ev?: unknown) => void) | null = null;
  onerror: ((ev?: unknown) => void) | null = null;
  onclose: ((ev?: unknown) => void) | null = null;
  close(): void {}
  constructor() {
    wsInstances.push(this);
  }
}

interface Deferred {
  id: string;
  offset: number;
  resolve: (v: { chunk: string; offset: number }) => void;
}

function mountWith(jobId: ReturnType<typeof ref<string>>) {
  let stream: ReturnType<typeof useJobLogStream> | null = null;
  const Host = defineComponent({
    setup() {
      stream = useJobLogStream(jobId);
      return () => null;
    },
  });
  const app = createApp(Host);
  app.mount(document.createElement("div"));
  return { stream: stream as unknown as ReturnType<typeof useJobLogStream>, app };
}

const flush = async () => {
  // Drain the microtask queue so pollHttp's awaited continuation runs.
  await Promise.resolve();
  await Promise.resolve();
};

describe("useJobLogStream", () => {
  let deferreds: Deferred[];

  beforeEach(() => {
    vi.useFakeTimers();
    (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWebSocket;
    wsInstances.length = 0;
    deferreds = [];
    jobLogs.mockImplementation((id: string, offset: number) => {
      return new Promise<{ chunk: string; offset: number }>((resolve) => {
        deferreds.push({ id, offset, resolve });
      });
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("discards a late HTTP response from a previous job after a job switch", async () => {
    const jobId = ref<string>("job-A");
    const { stream, app } = mountWith(jobId);

    // job-A's prime poll is in flight.
    expect(deferreds.map((d) => d.id)).toEqual(["job-A"]);

    // Switch to job-B before job-A's poll resolves.
    jobId.value = "job-B";
    await nextTick();
    expect(deferreds.map((d) => d.id)).toEqual(["job-A", "job-B"]);

    // The stale job-A response resolves now — it must NOT append to job-B's log.
    deferreds[0].resolve({ chunk: "OLD-A", offset: 999 });
    await flush();
    expect(stream.logText.value).toBe("");

    // job-B's response appends normally.
    deferreds[1].resolve({ chunk: "NEW-B", offset: 5 });
    await flush();
    expect(stream.logText.value).toBe("NEW-B");

    // The next poll continues from job-B's offset (5), proving the stale offset (999)
    // never clobbered it.
    vi.advanceTimersByTime(2000);
    await flush();
    const last = deferreds[deferreds.length - 1];
    expect(last.id).toBe("job-B");
    expect(last.offset).toBe(5);

    app.unmount();
  });

  it("stops polling once the socket connected and closed, even on an abnormal (1006) close", async () => {
    const jobId = ref<string>("job-T");
    const { stream, app } = mountWith(jobId);
    const ws = wsInstances[wsInstances.length - 1];

    // connect() fired a prime HTTP poll while the socket was opening.
    expect(deferreds.map((d) => d.id)).toEqual(["job-T"]);

    // The socket opens, delivers the full log, then the server closes it WITHOUT a clean handshake
    // (Starlette closes a finished job's logs socket with code 1006). Polling must not resume.
    ws.onopen?.();
    ws.onmessage?.({ data: "FULL LOG" });
    expect(stream.logText.value).toBe("FULL LOG");
    ws.onclose?.({ wasClean: false });

    const callsAtClose = deferreds.length;

    // The prime poll resolves AFTER the close — it must neither re-append the log nor reschedule.
    deferreds[0].resolve({ chunk: "FULL LOG", offset: 8920709 });
    await flush();
    vi.advanceTimersByTime(10000);
    await flush();

    expect(stream.logText.value).toBe("FULL LOG");
    expect(deferreds.length).toBe(callsAtClose); // no further GET /logs polls

    app.unmount();
  });

  it("falls back to HTTP polling only when the socket never connects", async () => {
    const jobId = ref<string>("job-D");
    const { app } = mountWith(jobId);
    const ws = wsInstances[wsInstances.length - 1];

    // No onopen — the socket never connected (WS-hostile env), then it errors/closes.
    ws.onclose?.({ wasClean: false });
    const before = deferreds.length;

    vi.advanceTimersByTime(2000);
    await flush();

    expect(deferreds.length).toBeGreaterThan(before); // fallback poll resumed

    app.unmount();
  });
});

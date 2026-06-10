import { createApp, defineComponent, nextTick, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { jobLogs } = vi.hoisted(() => ({ jobLogs: vi.fn() }));
vi.mock("../api", () => ({ api: { jobLogs } }));

import { useJobLogStream } from "./useJobLogStream";

// A WebSocket that never opens, so the composable stays on its HTTP-poll fallback path —
// which is exactly where the stale-response race lives.
class FakeWebSocket {
  onopen: ((ev?: unknown) => void) | null = null;
  onmessage: ((ev?: unknown) => void) | null = null;
  onerror: ((ev?: unknown) => void) | null = null;
  onclose: ((ev?: unknown) => void) | null = null;
  close(): void {}
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
});

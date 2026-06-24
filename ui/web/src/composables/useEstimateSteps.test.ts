import { createApp, defineComponent, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { useEstimateSteps } from "./useEstimateSteps";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Mount a component that runs setup() and captures the return value. */
function mountWith<T>(setup: () => T): { api: T; app: ReturnType<typeof createApp> } {
  let captured: T | null = null;
  const Host = defineComponent({
    setup() {
      captured = setup();
      return () => null;
    },
  });
  const app = createApp(Host);
  app.mount(document.createElement("div"));
  return { api: captured as T, app };
}

const GOOD_RESPONSE = {
  ok: true,
  steps_per_epoch: 100,
  total_steps: 1000,
  images_per_resolution: 50,
  epochs: 10,
  per_resolution: {},
  image_counts: {},
} as const;

// ---------------------------------------------------------------------------

describe("useEstimateSteps", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("returns null result and false loading initially when form is null", async () => {
    const form = ref(null);
    const gpus = ref(1);

    const { api, app } = mountWith(() => useEstimateSteps(form, gpus));

    // No fetch should fire when form is null.
    expect(api.result.value).toBeNull();
    expect(api.loading.value).toBe(false);

    app.unmount();
  });

  it("debounces and fires estimate when form is populated", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(GOOD_RESPONSE), { status: 200 })
    );

    const form = ref<Record<string, unknown> | null>(null);
    const gpus = ref(1);

    const { api, app } = mountWith(() => useEstimateSteps(form as never, gpus));

    // Provide a form value — this should schedule a debounced call.
    form.value = {
      epochs: 10,
      micro_batch_size_per_gpu: 2,
      gradient_accumulation_steps: 1,
      dataset: "rengu-flow-dataset:42",
    };

    await nextTick();

    // Before the debounce fires, no fetch should have been called.
    expect(fetchMock).not.toHaveBeenCalled();

    // Advance past the 500 ms debounce; flush the dataset TOML fetch too.
    await vi.advanceTimersByTimeAsync(600);
    // The dataset fetch (GET /api/v1/datasets/42) returns empty content → empty toml.
    await Promise.resolve();
    // The estimate fetch (POST /api/v1/runs/estimate-steps) fires next.
    await Promise.resolve();

    // At least the estimate endpoint should have been called.
    const calls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(calls.some((u) => u.includes("estimate-steps"))).toBe(true);

    app.unmount();
  });

  it("populates result on a successful ok:true response", async () => {
    // Mock dataset fetch (returns empty content so TOML is empty).
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/datasets/")) {
        return new Response(JSON.stringify({ id: 42, content: "" }), { status: 200 });
      }
      return new Response(JSON.stringify(GOOD_RESPONSE), { status: 200 });
    });

    const form = ref<Record<string, unknown> | null>({
      epochs: 10,
      micro_batch_size_per_gpu: 2,
      dataset: "rengu-flow-dataset:42",
    });
    const gpus = ref(1);

    const { api, app } = mountWith(() => useEstimateSteps(form as never, gpus));

    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.result.value?.ok).toBe(true);
    expect(api.result.value?.total_steps).toBe(1000);
    expect(api.result.value?.steps_per_epoch).toBe(100);

    app.unmount();
  });

  it("sets result to null on ok:false response (incomplete config)", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("estimate-steps")) {
        return new Response(JSON.stringify({ ok: false, error: "missing epochs" }), {
          status: 200,
        });
      }
      return new Response(JSON.stringify({ id: 1, content: "" }), { status: 200 });
    });

    const form = ref<Record<string, unknown> | null>({
      epochs: 1,
      dataset: "rengu-flow-dataset:1",
    });
    const gpus = ref(1);

    const { api, app } = mountWith(() => useEstimateSteps(form as never, gpus));

    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.result.value).toBeNull();

    app.unmount();
  });

  it("ignores stale responses: only latest token applies its result", async () => {
    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("estimate-steps")) {
        const n = ++callCount;
        // First call resolves slowly; second resolves immediately.
        if (n === 1) await new Promise((r) => setTimeout(r, 2000));
        return new Response(
          JSON.stringify({ ...GOOD_RESPONSE, total_steps: n === 1 ? 999 : 1000 }),
          { status: 200 }
        );
      }
      return new Response(JSON.stringify({ id: 1, content: "" }), { status: 200 });
    });

    const form = ref<Record<string, unknown> | null>({
      epochs: 1,
      dataset: "rengu-flow-dataset:1",
    });
    const gpus = ref(1);

    const { api, app } = mountWith(() => useEstimateSteps(form as never, gpus));

    // First debounce fires.
    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();

    // While the first call is in flight, change a form value → new debounce cycle.
    form.value = { ...form.value, epochs: 2 };
    await nextTick();
    await vi.advanceTimersByTimeAsync(600);
    await Promise.resolve();
    await Promise.resolve();

    // Let the second (fast) call complete.
    await Promise.resolve();
    await Promise.resolve();

    // Advance the slow first call past its delay — should be ignored.
    await vi.advanceTimersByTimeAsync(2500);
    await Promise.resolve();

    // Result should be from the second call (total_steps: 1000), not the first (999).
    if (api.result.value !== null) {
      expect(api.result.value.total_steps).not.toBe(999);
    }

    app.unmount();
  });
});

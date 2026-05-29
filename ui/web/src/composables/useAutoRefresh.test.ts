import { defineComponent, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";
import { useAutoRefresh } from "./useAutoRefresh";

describe("useAutoRefresh", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.setItem("rengu-flow.autoRefreshInterval", "10");
  });

  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  it("schedules polling after the initial manual refresh on mount", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    let api: ReturnType<typeof useAutoRefresh> | null = null;

    const Host = defineComponent({
      setup() {
        api = useAutoRefresh({ refresh, immediate: true, isActive: () => true });
        return () => null;
      },
    });

    const el = document.createElement("div");
    const app = createApp(Host);
    app.mount(el);
    await nextTick();
    await Promise.resolve();

    expect(refresh).toHaveBeenCalledTimes(1);

    refresh.mockClear();
    await vi.advanceTimersByTimeAsync(10_000);
    await Promise.resolve();

    expect(refresh).toHaveBeenCalledTimes(1);

    app.unmount();
    api = null;
  });

  it("refreshNow resets the countdown", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    let api: ReturnType<typeof useAutoRefresh> | null = null;

    const Host = defineComponent({
      setup() {
        api = useAutoRefresh({ refresh, immediate: true, isActive: () => true });
        return () => null;
      },
    });

    const el = document.createElement("div");
    const app = createApp(Host);
    app.mount(el);
    await Promise.resolve();

    refresh.mockClear();
    await vi.advanceTimersByTimeAsync(5_000);
    await api!.refreshNow();
    await Promise.resolve();

    refresh.mockClear();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(refresh).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(5_000);
    await Promise.resolve();
    expect(refresh).toHaveBeenCalledTimes(1);

    app.unmount();
  });

  it("restarts polling when the interval changes", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined);
    let api: ReturnType<typeof useAutoRefresh> | null = null;

    const Host = defineComponent({
      setup() {
        api = useAutoRefresh({ refresh, immediate: true, isActive: () => true });
        return () => null;
      },
    });

    const el = document.createElement("div");
    const app = createApp(Host);
    app.mount(el);
    await Promise.resolve();

    refresh.mockClear();
    api!.setIntervalSec(5);
    await Promise.resolve();

    refresh.mockClear();
    await vi.advanceTimersByTimeAsync(5_000);
    await Promise.resolve();

    expect(refresh).toHaveBeenCalledTimes(1);

    app.unmount();
  });
});

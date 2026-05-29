import { createApp, defineComponent, nextTick } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useLatestAsync } from "./useLatestAsync";

function mountWith(setup: () => unknown) {
  let api: ReturnType<typeof useLatestAsync> | null = null;
  const Host = defineComponent({
    setup() {
      api = setup() as ReturnType<typeof useLatestAsync>;
      return () => null;
    },
  });
  const app = createApp(Host);
  app.mount(document.createElement("div"));
  return { api: api as unknown as ReturnType<typeof useLatestAsync>, app };
}

describe("useLatestAsync", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("treats only the most recent token as current", () => {
    const { api, app } = mountWith(() => useLatestAsync());
    const first = api.begin();
    const second = api.begin();
    expect(api.isCurrent(first)).toBe(false);
    expect(api.isCurrent(second)).toBe(true);
    app.unmount();
  });

  it("debounces and replaces a pending scheduled call", () => {
    const { api, app } = mountWith(() => useLatestAsync());
    const fn = vi.fn();
    api.schedule(fn, 100);
    api.schedule(fn, 100); // replaces the first timer
    vi.advanceTimersByTime(100);
    expect(fn).toHaveBeenCalledTimes(1);
    app.unmount();
  });

  it("cancel invalidates in-flight work and pending timers", () => {
    const { api, app } = mountWith(() => useLatestAsync());
    const token = api.begin();
    const fn = vi.fn();
    api.schedule(fn, 100);
    api.cancel();
    vi.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();
    expect(api.isCurrent(token)).toBe(false);
    app.unmount();
  });

  it("clears a pending timer on unmount", () => {
    const { api, app } = mountWith(() => useLatestAsync());
    const fn = vi.fn();
    api.schedule(fn, 100);
    app.unmount();
    vi.advanceTimersByTime(200);
    expect(fn).not.toHaveBeenCalled();
  });
});

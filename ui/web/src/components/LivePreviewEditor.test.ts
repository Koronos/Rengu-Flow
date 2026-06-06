import { describe, expect, it, vi, beforeEach } from "vitest";
import { createApp, h, nextTick } from "vue";
import ElementPlus from "element-plus";

const getJobPreviewConfig = vi.fn();
const updateJobPreviewConfig = vi.fn();

vi.mock("../api", () => ({
  api: {
    getJobPreviewConfig: (...a: unknown[]) => getJobPreviewConfig(...a),
    updateJobPreviewConfig: (...a: unknown[]) => updateJobPreviewConfig(...a),
  },
}));

import LivePreviewEditor from "./LivePreviewEditor.vue";

function findButton(el: HTMLElement, exact: string): HTMLButtonElement | null {
  const buttons = Array.from(el.querySelectorAll("button"));
  return (
    (buttons.find((b) => (b.textContent ?? "").replace(/\s+/g, " ").trim() === exact) as
      | HTMLButtonElement
      | undefined) ?? null
  );
}

async function flush() {
  await nextTick();
  await new Promise((r) => setTimeout(r));
  await nextTick();
}

async function mountEditor() {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({ render: () => h(LivePreviewEditor, { jobId: "7" }) });
  app.use(ElementPlus);
  app.mount(el);
  await flush();
  return { el, unmount: () => (app.unmount(), el.remove()) };
}

describe("LivePreviewEditor", () => {
  beforeEach(() => {
    getJobPreviewConfig.mockReset();
    updateJobPreviewConfig.mockReset();
    updateJobPreviewConfig.mockResolvedValue({ ok: true, run_dir: "/x", preview_now: false });
  });

  it("loads current preview and preserves unknown keys on apply", async () => {
    getJobPreviewConfig.mockResolvedValue({
      preview: { prompts: ["old"], seed_stride: 3, enabled: true, preview_every_n_steps: 50 },
      active: true,
    });
    const { el, unmount } = await mountEditor();
    expect(getJobPreviewConfig).toHaveBeenCalledWith("7");

    findButton(el, "Apply")!.click();
    await flush();

    expect(updateJobPreviewConfig).toHaveBeenCalledTimes(1);
    const [jobId, preview, previewNow] = updateJobPreviewConfig.mock.calls[0];
    expect(jobId).toBe("7");
    expect(previewNow).toBe(false);
    expect(preview.prompts).toEqual(["old"]); // from the loaded prompts
    expect(preview.seed_stride).toBe(3); // unexposed key preserved
    expect(preview.enabled).toBe(true);
    expect(preview.preview_every_n_steps).toBe(50);
    unmount();
  });

  it("'Apply & preview now' requests an immediate preview", async () => {
    getJobPreviewConfig.mockResolvedValue({ preview: { prompts: ["a"] }, active: true });
    const { el, unmount } = await mountEditor();
    findButton(el, "Apply & preview now")!.click();
    await flush();
    expect(updateJobPreviewConfig).toHaveBeenCalledTimes(1);
    expect(updateJobPreviewConfig.mock.calls[0][2]).toBe(true);
    unmount();
  });
});

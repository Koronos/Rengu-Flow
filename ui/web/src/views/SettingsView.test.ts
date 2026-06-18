import { describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import ElementPlus from "element-plus";

// vi.mock is hoisted — no top-level variables from this module can be
// referenced inside the factory. Use vi.fn() inline; grab refs via import.
vi.mock("../api", () => ({
  api: {
    getSettings: vi.fn().mockResolvedValue({
      path: "/repo/rengu.local.toml",
      exists: true,
      editable: {
        training: { num_gpus: 1, master_port: 29500, extra_args: "", env: {} },
        maintenance: { enabled: false, allow_pip: false },
      },
      restartRequired: { ui: { public: false, token: null } },
      readOnly: {
        ui: { host: "127.0.0.1", port: 8765, data_dir: "data" },
        toolbox: { enabled: false },
      },
    }),
    updateSettings: vi.fn().mockResolvedValue({
      path: "/repo/rengu.local.toml",
      exists: true,
      editable: {
        training: { num_gpus: 1, master_port: 29500, extra_args: "", env: {} },
        maintenance: { enabled: false, allow_pip: false },
      },
      restartRequired: { ui: { public: false, token: null } },
      readOnly: {
        ui: { host: "127.0.0.1", port: 8765, data_dir: "data" },
        toolbox: { enabled: false },
      },
    }),
  },
}));

import { api } from "../api";
import SettingsView from "./SettingsView.vue";

/** Mount the view, wait for async load, return { app, el, instance }. */
async function mountSettingsView() {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp(SettingsView);
  app.use(ElementPlus);
  const instance = app.mount(el) as unknown as Record<string, unknown>;
  // Wait for the async getSettings call to resolve
  await new Promise((r) => setTimeout(r, 0));
  await nextTick();
  return { app, el, instance };
}

describe("SettingsView", () => {
  it("loads settings and renders the theme toggle", async () => {
    const { app, el } = await mountSettingsView();

    expect(el.querySelector(".theme-toggle")).toBeTruthy();
    expect(el.textContent ?? "").toContain("127.0.0.1");

    app.unmount();
    el.remove();
  });

  it("normalizes empty-string env (KeyValueListField cleared) to {} in the patch", async () => {
    vi.mocked(api.updateSettings).mockClear();
    const { app, instance } = await mountSettingsView();

    // defineExpose auto-unwraps refs, so instance.form is the settings object directly.
    // Simulate KeyValueListField emitting "" when the last row is cleared.
    const form = instance.form as { editable: { training: Record<string, unknown> } } | null;
    if (form) {
      form.editable.training.env = "";
    }

    const onSave = instance.onSave as () => Promise<void>;
    await onSave();

    expect(api.updateSettings).toHaveBeenCalledOnce();
    const patch = vi.mocked(api.updateSettings).mock.calls[0][0] as { training: { env: unknown } };
    expect(patch.training.env).toEqual({});

    app.unmount();
  });

  it("stringifies non-string env values in the patch", async () => {
    vi.mocked(api.updateSettings).mockClear();
    const { app, instance } = await mountSettingsView();

    // defineExpose auto-unwraps refs, so instance.form is the settings object directly.
    const form = instance.form as { editable: { training: Record<string, unknown> } } | null;
    if (form) {
      form.editable.training.env = { NUM: 42, FLAG: true };
    }

    const onSave = instance.onSave as () => Promise<void>;
    await onSave();

    expect(api.updateSettings).toHaveBeenCalledOnce();
    const patch = vi.mocked(api.updateSettings).mock.calls[0][0] as { training: { env: unknown } };
    expect(patch.training.env).toEqual({ NUM: "42", FLAG: "true" });

    app.unmount();
  });
});

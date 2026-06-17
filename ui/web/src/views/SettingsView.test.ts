import { describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import ElementPlus from "element-plus";

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
      readOnly: { ui: { host: "127.0.0.1", port: 8765, data_dir: "data" } },
    }),
    updateSettings: vi.fn(),
  },
}));

import SettingsView from "./SettingsView.vue";

describe("SettingsView", () => {
  it("loads settings and renders the theme toggle", async () => {
    const el = document.createElement("div");
    document.body.appendChild(el);

    const app = createApp(SettingsView);
    app.use(ElementPlus);
    app.mount(el);

    // Wait for the async getSettings call to resolve
    await new Promise((r) => setTimeout(r, 0));
    await nextTick();

    expect(el.querySelector(".theme-toggle")).toBeTruthy();
    expect(el.textContent ?? "").toContain("127.0.0.1");

    app.unmount();
    el.remove();
  });
});

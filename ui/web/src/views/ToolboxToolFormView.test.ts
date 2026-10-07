/**
 * Editing a tool must not wipe the `io` it declared: the form seeds the declared values and saves
 * them back, and a tool that declared nothing keeps declaring nothing.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import ElementPlus from "element-plus";

vi.mock("vue-router", () => ({
  useRoute: () => ({ params: { id: "extract-images" } }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("../components/CodeEditor.vue", () => ({ default: { render: () => null } }));
vi.mock("../components/ToolboxRunPanel.vue", () => ({ default: { render: () => null } }));
vi.mock("../api", () => ({
  api: { getToolboxTool: vi.fn(), updateToolboxTool: vi.fn(async () => ({})) },
}));

import { api } from "../api";
import ToolboxToolFormView from "./ToolboxToolFormView.vue";

function tool(io: Record<string, unknown>) {
  return {
    id: "extract-images",
    name: "Extract images",
    description: "",
    entrypoint: "run",
    requirements: [],
    inputs: [],
    script: "def run():\n    return '/frames'\n",
    created_at: "",
    updated_at: "",
    last_run: null,
    io,
  };
}

async function openAndSave(io: Record<string, unknown>) {
  vi.mocked(api.getToolboxTool).mockResolvedValue(tool(io) as never);
  vi.mocked(api.updateToolboxTool).mockClear();
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp(ToolboxToolFormView);
  app.use(ElementPlus);
  app.mount(el);
  for (let i = 0; i < 6; i += 1) await new Promise((r) => setTimeout(r, 0));
  await nextTick();
  const save = [...document.querySelectorAll<HTMLElement>("button")].find((b) =>
    (b.textContent ?? "").trim() === "Save",
  );
  save?.click();
  for (let i = 0; i < 4; i += 1) await nextTick();
  app.unmount();
  return vi.mocked(api.updateToolboxTool).mock.calls[0]?.[1];
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("ToolboxToolFormView io", () => {
  it("saves back what the tool declared", async () => {
    const body = await openAndSave({
      input: "none",
      output: "folder",
      input_declared: true,
      output_declared: true,
    });
    expect(body?.io).toEqual({ input: "none", output: "folder" });
  });

  it("does not turn resolved defaults into declarations", async () => {
    const body = await openAndSave({
      input: "folder",
      output: "passthrough",
      input_declared: false,
      output_declared: false,
    });
    expect(body?.io).toEqual({});
  });

  it("keeps a half declaration half declared", async () => {
    const body = await openAndSave({
      input: "none",
      output: "passthrough",
      input_declared: true,
      output_declared: false,
    });
    expect(body?.io).toEqual({ input: "none" });
  });
});

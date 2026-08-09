/**
 * The safety net for the `ToolboxInputsForm` / `ToolboxLogPanel` extraction: the Toolbox page's
 * run panel must still render a tool's declared inputs and the finished run's console, and must
 * still send the seeded values when Run is pressed.
 */
import { describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import ElementPlus from "element-plus";
import ToolboxRunPanel from "./ToolboxRunPanel.vue";

/** The ANSI escape the log panel strips, spelled out rather than pasted as a control byte. */
const ESC = String.fromCharCode(27);

const tool = {
  id: "t1",
  name: "Resize",
  description: "",
  entrypoint: "run",
  requirements: [],
  script: "",
  created_at: "",
  updated_at: "",
  inputs: [
    { param: "size", label: "Longest side", control: "number", default: 1024, hint: "in pixels" },
    { param: "dry_run", label: "Dry run", control: "switch", default: false },
  ],
  last_run: { status: "done", exit_code: 0, inputs: { size: 768 } },
};

const runToolboxTool = vi.fn(async () => tool.last_run);

vi.mock("../api", () => ({
  api: {
    getToolboxTool: vi.fn(async () => tool),
    toolboxEnabled: vi.fn(async () => ({ enabled: true })),
    toolboxLog: vi.fn(async () => ({
      chunk: `${ESC}[32mhello from the tool${ESC}[0m\n`,
      offset: 12,
      status: "done",
    })),
    toolboxRunStatus: vi.fn(async () => ({ status: "done", exit_code: 0 })),
    runToolboxTool: (...args: unknown[]) => runToolboxTool(...(args as [])),
    cancelToolboxRun: vi.fn(async () => ({ ok: true })),
  },
}));

async function mountPanel() {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const errors: unknown[] = [];
  const app = createApp(ToolboxRunPanel, { toolId: "t1" });
  app.use(ElementPlus);
  app.config.errorHandler = (err) => {
    errors.push(err);
  };
  app.mount(el);
  for (let i = 0; i < 8; i += 1) await nextTick();
  return { app, el, errors };
}

describe("ToolboxRunPanel", () => {
  it("renders the declared inputs and the last run's console", async () => {
    const { app, el, errors } = await mountPanel();

    expect(errors).toEqual([]);
    // inputs, from ToolboxInputsForm
    expect(el.textContent).toContain("Longest side");
    expect(el.textContent).toContain("Dry run");
    expect(el.textContent).toContain("in pixels");
    // console, from ToolboxLogPanel — ANSI stripped, exit code shown
    expect(el.querySelector(".output__log")?.textContent).toBe("hello from the tool\n");
    expect(el.querySelector(".output__meta")?.textContent).toContain("exit 0");

    app.unmount();
    el.remove();
  });

  it("sends the seeded values when Run is pressed", async () => {
    const { app, el, errors } = await mountPanel();
    expect(errors).toEqual([]);

    const runButton = Array.from(el.querySelectorAll("button")).find((b) =>
      (b.textContent || "").includes("Run"),
    );
    expect(runButton).toBeTruthy();
    runButton?.dispatchEvent(new Event("click"));
    await nextTick();

    // defaults from `inputs`, overridden by the last run's values
    expect(runToolboxTool).toHaveBeenCalledWith("t1", { size: 768, dry_run: false });

    app.unmount();
    el.remove();
  });
});

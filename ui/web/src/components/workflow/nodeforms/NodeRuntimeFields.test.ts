/**
 * The From select: any step may be a source (even one below), the ones that cannot say why, and a
 * source below asks the page to reorder instead of silently doing nothing.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createApp, h, nextTick, ref } from "vue";
import ElementPlus from "element-plus";
import NodeRuntimeFields from "./NodeRuntimeFields.vue";
import type { ToolIoMap } from "../../../lib/workflowNodeTypes";
import type { WorkflowGraph, WorkflowNode } from "../../../types/workflow";

vi.mock("../../../composables/useSystemStatsStream", () => ({
  useSystemStatsStream: () => ({ stats: { value: null } }),
}));

function node(id: string, type: string, from: string | null, config = {}): WorkflowNode {
  return {
    id,
    type,
    title: id,
    from,
    enabled: true,
    config,
    gpu: { required: false, wait: true, device: null },
  };
}

/** f -> tag (reads f) -> quality (reads tag); plus an independent tool x. */
function graph(): WorkflowGraph {
  return {
    version: 1,
    name: "",
    description: "",
    variables: [],
    nodes: [
      node("f", "folder", null),
      node("tag", "prep.tag", "f"),
      node("q", "prep.quality", "tag"),
      node("x", "tool", null, { tool_id: "extract" }),
    ],
  };
}

async function mountFor(nodeId: string, toolIo: ToolIoMap = {}) {
  const g = graph();
  const current = ref(g.nodes.find((n) => n.id === nodeId) as WorkflowNode);
  const updates: WorkflowNode[] = [];
  const repoints: (string | null)[] = [];
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({
    render: () =>
      h(NodeRuntimeFields, {
        modelValue: current.value,
        "onUpdate:modelValue": (value: WorkflowNode) => {
          updates.push(value);
          current.value = value;
        },
        onRepoint: (sourceId: string | null) => repoints.push(sourceId),
        graph: g,
        toolIo,
        sourcePaths: { f: "D:/data", x: "a new folder, decided when this step runs" },
      }),
  });
  app.use(ElementPlus);
  app.mount(el);
  for (let i = 0; i < 6; i += 1) await nextTick();
  return { app, updates, repoints };
}

/** Open the From select and return its options (the dropdown is teleported to <body>). */
async function openFrom(): Promise<HTMLElement[]> {
  document.querySelector<HTMLElement>(".el-select__wrapper")?.click();
  for (let i = 0; i < 6; i += 1) await nextTick();
  return [...document.querySelectorAll<HTMLElement>(".el-select-dropdown__item")];
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("NodeRuntimeFields — From", () => {
  it("offers a step below, marked, and keeps the steps that would loop out with their reason", async () => {
    const { app } = await mountFor("tag");
    const options = await openFrom();
    const labels = options.map((o) => o.textContent ?? "");

    expect(labels.some((l) => l.includes("x") && l.includes("below"))).toBe(true); // the tool, below
    // quality reads from tag, so it cannot be tag's source: listed disabled, with why.
    const loop = options.find((o) => (o.textContent ?? "").includes("loop"));
    expect(loop?.classList.contains("is-disabled")).toBe(true);
    expect(document.body.textContent).toContain("Not offered");

    app.unmount();
  });

  it("asks the page to reorder when the chosen source is below, and just re-points otherwise", async () => {
    const { app, updates, repoints } = await mountFor("tag");
    const options = await openFrom();

    options.find((o) => (o.textContent ?? "").includes("below"))?.click(); // the tool x
    for (let i = 0; i < 4; i += 1) await nextTick();
    expect(repoints).toEqual(["x"]);
    expect(updates).toEqual([]); // no in-place `from` edit: the graph is reordered by the page

    app.unmount();
  });

  it("names a refused tool source and why", async () => {
    const toolIo: ToolIoMap = { extract: { input: "none", output: "none", output_declared: true } };
    const { app } = await mountFor("tag", toolIo);
    const options = await openFrom();

    const tool = options.find((o) => (o.textContent ?? "").includes("outputs nothing"));
    expect(tool?.classList.contains("is-disabled")).toBe(true);

    app.unmount();
  });
});

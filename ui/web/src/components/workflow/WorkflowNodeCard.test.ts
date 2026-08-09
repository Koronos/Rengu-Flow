import { describe, expect, it } from "vitest";
import { createApp, nextTick, h } from "vue";
import ElementPlus from "element-plus";
import WorkflowNodeCard from "./WorkflowNodeCard.vue";
import { nodeChip } from "../../lib/workflowStatus";
import type { NodeState, WorkflowNode } from "../../types/workflow";

function baseNode(over: Partial<WorkflowNode> = {}): WorkflowNode {
  return {
    id: "n2",
    type: "prep.tag",
    title: "Tag",
    from: "n1",
    enabled: true,
    config: {},
    gpu: { required: true, wait: true, device: 0 },
    ...over,
  };
}

async function mountCard(props: Record<string, unknown>) {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({
    render: () =>
      h(WorkflowNodeCard, {
        position: 2,
        summary: "pixai-v0.9 · GPU 0",
        outputSentence: "Writes tag sidecars into the input folder and emits it unchanged",
        ...props,
      } as never),
  });
  app.use(ElementPlus);
  app.mount(el);
  await nextTick();
  return {
    el,
    unmount() {
      app.unmount();
      el.remove();
    },
  };
}

function chipFor(entry: NodeState | undefined, node = baseNode()) {
  return nodeChip(node, entry);
}

describe("WorkflowNodeCard", () => {
  it("draws no jump badge for a consecutive link — the connector already says it", async () => {
    const card = await mountCard({ node: baseNode(), chip: chipFor(undefined), jump: null });
    expect(card.el.querySelector(".wf-card__badge")).toBeNull();
    card.unmount();
  });

  it("draws the badge, and only the badge, for a jump", async () => {
    const card = await mountCard({
      node: baseNode(),
      chip: chipFor(undefined),
      jump: { key: "n1->n2", sourcePosition: 1 },
    });
    const badge = card.el.querySelector(".wf-card__badge");
    expect(badge).not.toBeNull();
    expect(badge?.textContent?.trim()).toBe("⟵ from ①");
    // A button, so hover-highlight and Enter-to-source work from the keyboard.
    expect(badge?.tagName).toBe("BUTTON");
    card.unmount();
  });

  it("paints done AND stale at the same time", async () => {
    const card = await mountCard({
      node: baseNode(),
      chip: chipFor({ status: "done" }),
      stale: true,
    });
    const cardEl = card.el.querySelector(".wf-card");
    expect(cardEl?.classList.contains("wf-card--stale")).toBe(true);
    expect(card.el.querySelector(".wf-card__chip")?.textContent).toContain("Done");
    expect(card.el.querySelector(".wf-card__stale")?.textContent).toContain("stale");
    card.unmount();
  });

  it("strikes a disabled step out", async () => {
    const node = baseNode({ enabled: false });
    const card = await mountCard({ node, chip: chipFor({ status: "done" }, node) });
    expect(card.el.querySelector(".wf-card")?.classList.contains("wf-card--disabled")).toBe(true);
    card.unmount();
  });

  it("shows a progress bar and a clamped percentage only while running", async () => {
    const running = await mountCard({
      node: baseNode(),
      chip: chipFor({ status: "running" }),
      percent: 41.4,
    });
    expect(running.el.querySelector(".wf-card__progress")).not.toBeNull();
    expect(running.el.querySelector(".wf-card__percent")?.textContent).toContain("41%");
    running.unmount();

    const idle = await mountCard({ node: baseNode(), chip: chipFor(undefined) });
    expect(idle.el.querySelector(".wf-card__progress")).toBeNull();
    idle.unmount();
  });

  it("shows the first line of a failure under the card", async () => {
    const card = await mountCard({
      node: baseNode(),
      chip: chipFor({ status: "failed", error: "Dataset folder not found: D:/nope\nTraceback…" }),
    });
    expect(card.el.querySelector(".wf-card__detail")?.textContent).toContain(
      "Dataset folder not found: D:/nope"
    );
    card.unmount();
  });

  it("renders the skipped-step legend when a hovered jump reads past it", async () => {
    const card = await mountCard({
      node: baseNode(),
      chip: chipFor(undefined),
      skippedNote: "③ reads past this step",
    });
    expect(card.el.querySelector(".wf-card__legend")?.textContent).toContain(
      "③ reads past this step"
    );
    card.unmount();
  });
});

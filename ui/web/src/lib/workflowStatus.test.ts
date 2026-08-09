import { describe, expect, it } from "vitest";
import {
  clockTime,
  firstLine,
  isBusy,
  moveBlockReason,
  nodeChip,
  nodesToRun,
  runFromBlockReason,
  shouldRunNode,
} from "./workflowStatus";
import type { NodeState, WorkflowGraph, WorkflowNode } from "../types/workflow";

function node(id: string, over: Partial<WorkflowNode> = {}): WorkflowNode {
  return {
    id,
    type: "prep.tag",
    title: id,
    from: null,
    enabled: true,
    config: {},
    gpu: { required: false, wait: true, device: null },
    ...over,
  };
}

function graph(...nodes: WorkflowNode[]): WorkflowGraph {
  return { version: 1, name: "wf", description: "", variables: [], nodes };
}

describe("isBusy", () => {
  it("covers exactly the statuses that make the editor read-only", () => {
    expect(isBusy("running")).toBe(true);
    expect(isBusy("cancelling")).toBe(true);
    expect(isBusy("idle")).toBe(false);
    expect(isBusy("failed")).toBe(false);
    expect(isBusy(undefined)).toBe(false);
  });
});

describe("firstLine", () => {
  it("takes the first non-empty line", () => {
    expect(firstLine("\n\n  boom  \nsecond")).toBe("boom");
    expect(firstLine("")).toBe("");
    expect(firstLine(undefined)).toBe("");
  });
});

describe("clockTime", () => {
  it("returns empty for anything unparseable", () => {
    expect(clockTime("not a date")).toBe("");
    expect(clockTime(null)).toBe("");
  });

  it("formats a real timestamp as HH:MM", () => {
    expect(clockTime(new Date(2026, 7, 9, 4, 7).toISOString())).toBe("04:07");
  });
});

describe("nodeChip", () => {
  it("strikes a disabled node out regardless of its saved status", () => {
    const chip = nodeChip(node("n1", { enabled: false }), { status: "done" });
    expect(chip.label).toBe("Disabled");
  });

  it("reads pending as Queued under a running workflow and Not run otherwise", () => {
    expect(nodeChip(node("n1"), undefined, "running").label).toBe("Queued");
    expect(nodeChip(node("n1"), undefined, "idle").label).toBe("Not run");
  });

  it("carries the GPU wait reason as the detail", () => {
    const chip = nodeChip(node("n1"), { status: "waiting_gpu", error: "held by run #42" });
    expect(chip.tone).toBe("warning");
    expect(chip.detail).toBe("held by run #42");
  });

  it("shows only the first line of a failure", () => {
    const chip = nodeChip(node("n1"), { status: "failed", error: "boom\nTraceback…" });
    expect(chip.detail).toBe("boom");
    expect(chip.tone).toBe("danger");
  });

  it("prefers a train node's queued run over the finish time", () => {
    const chip = nodeChip(node("n1", { type: "train" }), {
      status: "done",
      finished_at: new Date(2026, 7, 9, 4, 7).toISOString(),
      result: { job_id: 123 },
    });
    expect(chip.detail).toBe("Queued run #123");
  });

  it("only asks for a progress bar while something is actually moving", () => {
    const statuses: NodeState["status"][] = ["running", "launching", "stopping"];
    for (const status of statuses) {
      expect(nodeChip(node("n1"), { status }).showProgress).toBe(true);
    }
    expect(nodeChip(node("n1"), { status: "done" }).showProgress).toBe(false);
  });
});

describe("shouldRunNode / nodesToRun", () => {
  it("skips a done-and-fresh node and takes a done-and-stale one", () => {
    expect(shouldRunNode(node("n1"), { status: "done" }, false)).toBe(false);
    expect(shouldRunNode(node("n1"), { status: "done" }, true)).toBe(true);
  });

  it("includes stopped — resuming is the whole point", () => {
    expect(shouldRunNode(node("n1"), { status: "stopped" }, false)).toBe(true);
  });

  it("includes failed and never-run nodes, excludes disabled ones", () => {
    expect(shouldRunNode(node("n1"), { status: "failed" }, false)).toBe(true);
    expect(shouldRunNode(node("n1"), undefined, false)).toBe(true);
    expect(shouldRunNode(node("n1", { enabled: false }), undefined, true)).toBe(false);
  });

  it("returns the run set in list order", () => {
    const g = graph(node("n1"), node("n2"), node("n3"), node("n4", { enabled: false }));
    const state = {
      nodes: {
        n1: { status: "done" as const },
        n2: { status: "done" as const },
        n3: { status: "failed" as const },
      },
    };
    expect(nodesToRun(g, state, { n2: true })).toEqual(["n2", "n3"]);
  });
});

describe("runFromBlockReason", () => {
  const g = graph(node("n1", { type: "folder" }), node("n2", { from: "n1" }), node("n3", { from: "n2" }));

  it("allows starting at a node whose source has a saved output", () => {
    const state = { nodes: { n2: { output: { path: "D:/a", caption_format: "sidecar", caption_ext: ".txt" } } } };
    expect(runFromBlockReason(g, "n3", state)).toBe("");
  });

  it("names the source and a safe restart point when there is no saved output", () => {
    expect(runFromBlockReason(g, "n3", { nodes: {} })).toBe(
      "② has no saved output. Start from ① or earlier."
    );
  });

  it("points at Run when the missing output is the very first node", () => {
    expect(runFromBlockReason(g, "n2", { nodes: {} })).toBe(
      "① has no saved output. Use Run to start from the top."
    );
  });

  it("never blocks a sourceless node", () => {
    expect(runFromBlockReason(g, "n1", { nodes: {} })).toBe("");
  });
});

describe("moveBlockReason", () => {
  const g = graph(node("n1", { type: "folder" }), node("n2", { from: "n1" }), node("n3", { from: "n2" }));

  it("is empty when the move is legal", () => {
    const detached = graph(node("a", { type: "folder" }), node("b", { type: "folder" }));
    expect(moveBlockReason(detached, "b", "up")).toBe("");
  });

  it("tells the user to lift the source first", () => {
    expect(moveBlockReason(g, "n3", "up")).toBe("③ reads from ②. Move ② up first.");
  });

  it("mirrors the message for the equivalent downward move", () => {
    expect(moveBlockReason(g, "n2", "down")).toBe("③ reads from ②. Move ③ down first.");
  });

  it("says so at the ends of the list", () => {
    expect(moveBlockReason(g, "n1", "up")).toBe("Already the first step.");
    expect(moveBlockReason(g, "n3", "down")).toBe("Already the last step.");
  });
});

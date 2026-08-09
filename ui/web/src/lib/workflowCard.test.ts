import { describe, expect, it } from "vitest";
import { chainSummary, nodeConfigSummary, relativeTime, workflowResultPath } from "./workflowCard";
import type { WorkflowNode } from "../types/workflow";

function node(over: Partial<WorkflowNode> = {}): WorkflowNode {
  return {
    id: "n1",
    type: "folder",
    title: "Source folder",
    from: null,
    enabled: true,
    config: {},
    gpu: { required: false, wait: true, device: null },
    ...over,
  };
}

describe("nodeConfigSummary", () => {
  it("reads a folder as path plus caption layout", () => {
    const summary = nodeConfigSummary(
      node({ config: { path: "D:/datasets/aoi", caption_format: "sidecar", caption_ext: ".txt" } })
    );
    expect(summary).toBe("D:/datasets/aoi · sidecar .txt · CPU");
  });

  it("says so when the source folder is still empty", () => {
    expect(nodeConfigSummary(node())).toContain("no folder yet");
  });

  it("lists taggers and the tag cap", () => {
    const summary = nodeConfigSummary(
      node({
        type: "prep.tag",
        config: { models: ["pixai-v0.9", "cl-tagger-1.02"], max_tags: 255 },
        gpu: { required: true, wait: true, device: 0 },
      })
    );
    expect(summary).toBe("pixai-v0.9 + cl-tagger-1.02 · max 255 · GPU 0");
  });

  it("shows the blur threshold only for the blur metric", () => {
    const blur = nodeConfigSummary(
      node({ type: "prep.quality", config: { metric: "blur", blur_threshold: 80, action: "move" } })
    );
    expect(blur).toBe("blur · blur < 80 · quarantine flagged · CPU");

    const iqa = nodeConfigSummary(
      node({ type: "prep.quality", config: { metric: "iqa", iqa_model: "clipiqa" } })
    );
    expect(iqa).toBe("iqa · clipiqa · report only · CPU");
  });

  it("surfaces clean's copy_undetected hazard", () => {
    const summary = nodeConfigSummary(
      node({ type: "prep.clean", config: { in_place: false, copy_undetected: false } })
    );
    expect(summary).toContain("undetected images left behind");
  });

  it("reads a train node as the run it will queue", () => {
    expect(nodeConfigSummary(node({ type: "train", config: { job_id: 42 } }))).toContain("run #42");
    expect(nodeConfigSummary(node({ type: "train", config: {} }))).toContain("no run selected");
  });

  it("does not pretend to understand a type from a newer app", () => {
    expect(nodeConfigSummary(node({ type: "prep.upscale" }))).toContain("unknown step type");
  });

  it("labels an auto-device GPU node without inventing an index", () => {
    const summary = nodeConfigSummary(
      node({ type: "prep.index", config: { models: ["a"] }, gpu: { required: true, wait: true, device: null } })
    );
    expect(summary).toBe("a · GPU");
  });
});

describe("chainSummary", () => {
  it("joins type labels with arrows", () => {
    expect(chainSummary(["folder", "prep.tag"])).toBe("Source folder → Tag");
  });

  it("falls back to the raw type for one it does not recognize", () => {
    expect(chainSummary(["prep.upscale"])).toBe("prep.upscale");
  });

  it("truncates a long chain", () => {
    const chain = ["folder", "prep.tag", "prep.caption", "prep.clean", "prep.quality", "tool"];
    expect(chainSummary(chain, 3)).toBe("Source folder → Tag → Caption → +3 more");
  });

  it("has something to say about an empty workflow", () => {
    expect(chainSummary([])).toBe("No steps yet");
  });
});

describe("workflowResultPath", () => {
  const handle = (path: string) => ({ path, caption_format: "sidecar", caption_ext: ".txt" });

  it("takes the last enabled node that saved an output", () => {
    const nodes = [node({ id: "n1" }), node({ id: "n2" }), node({ id: "n3", type: "train" })];
    const state = { nodes: { n1: { output: handle("D:/a") }, n2: { output: handle("D:/b") } } };
    expect(workflowResultPath({ nodes }, state)).toBe("D:/b");
  });

  it("skips a disabled tail", () => {
    const nodes = [node({ id: "n1" }), node({ id: "n2", enabled: false })];
    const state = { nodes: { n1: { output: handle("D:/a") }, n2: { output: handle("D:/b") } } };
    expect(workflowResultPath({ nodes }, state)).toBe("D:/a");
  });

  it("is empty before anything has run", () => {
    expect(workflowResultPath({ nodes: [node()] }, {})).toBe("");
  });
});

describe("relativeTime", () => {
  const now = Date.parse("2026-08-09T12:00:00+00:00");

  it("formats the usual buckets", () => {
    expect(relativeTime("2026-08-09T11:59:30+00:00", now)).toBe("just now");
    expect(relativeTime("2026-08-09T11:45:00+00:00", now)).toBe("15m ago");
    expect(relativeTime("2026-08-09T10:00:00+00:00", now)).toBe("2h ago");
    expect(relativeTime("2026-08-06T12:00:00+00:00", now)).toBe("3d ago");
  });

  it("clamps a clock-skewed future stamp instead of showing a negative age", () => {
    expect(relativeTime("2026-08-09T13:00:00+00:00", now)).toBe("just now");
  });

  it("is empty for junk", () => {
    expect(relativeTime("", now)).toBe("");
    expect(relativeTime("nope", now)).toBe("");
    expect(relativeTime(null, now)).toBe("");
  });
});

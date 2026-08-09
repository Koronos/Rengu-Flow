import { describe, expect, it } from "vitest";
import {
  NODE_TYPES,
  NODE_TYPE_GROUPS,
  NODE_TYPE_LIST,
  consumesInput,
  defaultNeedsGpu,
  describeOutput,
  emitsHandle,
  nodeTypeIcon,
  nodeTypeLabel,
  sourceMayBeEmpty,
} from "./workflowNodeTypes";

describe("the catalog", () => {
  it("carries the eight types of the spec, once each", () => {
    expect(NODE_TYPE_LIST.map((spec) => spec.type)).toEqual([
      "folder",
      "prep.tag",
      "prep.caption",
      "prep.clean",
      "prep.quality",
      "prep.index",
      "tool",
      "train",
    ]);
    expect(Object.keys(NODE_TYPES)).toHaveLength(8);
  });

  it("keeps the handle flags in parity with workflow_graph.py::NODE_TYPES", () => {
    // consumes / emits / sourceOptional, in the order of the backend catalog.
    expect(
      NODE_TYPE_LIST.map((s) => [s.type, s.consumes, s.emits, s.sourceOptional]),
    ).toEqual([
      ["folder", false, true, true],
      ["prep.tag", true, true, false],
      ["prep.caption", true, true, false],
      ["prep.clean", true, true, false],
      ["prep.quality", true, true, false],
      ["prep.index", true, true, false],
      ["tool", true, true, true],
      ["train", true, false, false],
    ]);
  });

  it("groups every type for the add menu", () => {
    const grouped = NODE_TYPE_GROUPS.flatMap((group) => group.types.map((spec) => spec.type));
    expect(new Set(grouped)).toEqual(new Set(NODE_TYPE_LIST.map((spec) => spec.type)));
    expect(NODE_TYPE_GROUPS.map((group) => group.id)).toEqual([
      "source",
      "prepare",
      "tools",
      "training",
    ]);
    expect(NODE_TYPE_GROUPS[1].types.map((spec) => spec.label)).toEqual([
      "Tag",
      "Caption",
      "Clean",
      "Quality filter",
      "Quality index",
    ]);
  });

  it("falls back on an unknown type instead of throwing", () => {
    expect(nodeTypeLabel("prep.fromTheFuture")).toBe("prep.fromTheFuture");
    expect(nodeTypeIcon("prep.fromTheFuture")).toBe("QuestionFilled");
    // Permissive: an unrecognised node keeps its links when an older app opens the graph.
    expect(consumesInput("prep.fromTheFuture")).toBe(true);
    expect(emitsHandle("prep.fromTheFuture")).toBe(true);
    expect(sourceMayBeEmpty("prep.fromTheFuture")).toBe(true);
  });

  it("knows which types may have no source", () => {
    expect(sourceMayBeEmpty("folder")).toBe(true);
    expect(sourceMayBeEmpty("tool")).toBe(true);
    expect(sourceMayBeEmpty("prep.clean")).toBe(false);
    expect(sourceMayBeEmpty("train")).toBe(false);
  });

  it("says only train is terminal", () => {
    expect(emitsHandle("train")).toBe(false);
    expect(emitsHandle("prep.quality")).toBe(true);
    expect(consumesInput("folder")).toBe(false);
  });
});

describe("defaultNeedsGpu", () => {
  it("follows the metric for prep.quality", () => {
    expect(defaultNeedsGpu("prep.quality", {})).toBe(false);
    expect(defaultNeedsGpu("prep.quality", { metric: "blur" })).toBe(false);
    expect(defaultNeedsGpu("prep.quality", { metric: "aesthetic" })).toBe(true);
    expect(defaultNeedsGpu("prep.quality", { metric: "iqa" })).toBe(true);
  });

  it("uses the catalog default for every other type", () => {
    expect(defaultNeedsGpu("prep.tag")).toBe(true);
    expect(defaultNeedsGpu("prep.caption")).toBe(true);
    expect(defaultNeedsGpu("prep.clean")).toBe(true);
    expect(defaultNeedsGpu("prep.index")).toBe(true);
    expect(defaultNeedsGpu("folder")).toBe(false);
    expect(defaultNeedsGpu("tool")).toBe(false);
    // train takes no lease itself; the queued job does.
    expect(defaultNeedsGpu("train")).toBe(false);
  });

  it("never demands a GPU for a type it does not know", () => {
    expect(defaultNeedsGpu("prep.fromTheFuture", { metric: "iqa" })).toBe(false);
  });
});

describe("describeOutput", () => {
  it("says the in-place stages emit their input unchanged", () => {
    expect(describeOutput({ type: "prep.tag" })).toBe(
      "Writes tag sidecars into the input folder and emits it unchanged",
    );
    expect(describeOutput({ type: "prep.caption" })).toContain("emits it unchanged");
    expect(describeOutput({ type: "prep.index" })).toContain("emits the input folder unchanged");
  });

  it("says quality emits its INPUT — its output_dir is the quarantine pile", () => {
    // The spec's output_dir trap: reading report["output_dir"] here would caption the rejects.
    expect(describeOutput({ type: "prep.quality", config: { output_dir: "D:/x/low_quality" } })).toBe(
      "Emits the input folder; flagged images are moved to the quarantine folder",
    );
  });

  it("is the only stage whose output_dir names the result: clean", () => {
    expect(describeOutput({ type: "prep.clean" })).toBe("Emits <input>/cleaned");
    expect(describeOutput({ type: "prep.clean", config: { output_dir: " D:/x/out " } })).toBe(
      "Emits D:/x/out",
    );
    expect(describeOutput({ type: "prep.clean", config: { in_place: true, output_dir: "D:/x/out" } })).toBe(
      "Emits the input folder; images are cleaned in place",
    );
  });

  it("covers folder, tool and the terminal train", () => {
    expect(describeOutput({ type: "folder" })).toContain("source");
    expect(describeOutput({ type: "tool" })).toContain("returns");
    expect(describeOutput({ type: "train" })).toBe("Emits nothing; training is the end of the chain");
  });

  it("names an unknown type instead of pretending it emits something", () => {
    expect(describeOutput({ type: "prep.fromTheFuture" })).toContain("prep.fromTheFuture");
    expect(describeOutput({ type: "prep.fromTheFuture" })).toContain("cannot run");
  });
});

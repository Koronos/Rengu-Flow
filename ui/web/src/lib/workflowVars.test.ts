import { describe, expect, it } from "vitest";
import type { WorkflowGraph, WorkflowNode } from "./workflowGraph";
import { collectRefs, resolveText, resolveValue, unknownRefs, variableMap } from "./workflowVars";

function node(id: string, type: string, config: Record<string, unknown>): WorkflowNode {
  return {
    id,
    type,
    title: id,
    from: null,
    enabled: true,
    config,
    gpu: { required: false, wait: true, device: null },
  };
}

// The cases below mirror tests/test_workflow_graph.py one for one; this is the only place the
// client duplicates server logic, so the two suites must keep saying the same thing.
describe("resolveText", () => {
  it("substitutes known names", () => {
    expect(resolveText("${a}/sub", { a: "D:/x" })).toBe("D:/x/sub");
    expect(resolveText("${a}${a}", { a: "z" })).toBe("zz");
  });

  it("escapes $$ to a literal $", () => {
    expect(resolveText("$$", {})).toBe("$");
    expect(resolveText("$${a}", { a: "z" })).toBe("${a}");
    expect(resolveText("cost: $$5 and ${a}", { a: "z" })).toBe("cost: $5 and z");
  });

  it("is a single pass with no recursion", () => {
    expect(resolveText("${a}", { a: "${b}", b: "deep" })).toBe("${b}");
  });

  it("leaves a missing variable literal, never emptied", () => {
    expect(resolveText("${nope}/x", { a: "1" })).toBe("${nope}/x");
  });

  it("ignores malformed tokens", () => {
    expect(resolveText("${1bad} $notavar {a}", { a: "z" })).toBe("${1bad} $notavar {a}");
    expect(resolveText("${}", { a: "z" })).toBe("${}");
    expect(resolveText("${a b}", { a: "z" })).toBe("${a b}");
  });

  it("inserts a value verbatim, without re-reading $ patterns in it", () => {
    // `$&` / `$1` are replacement directives to String.replace; a function replacement is immune.
    expect(resolveText("${a}", { a: "$& $1 $$" })).toBe("$& $1 $$");
  });

  it("accepts the graph's variable list as well as a plain map", () => {
    expect(resolveText("${a}", [{ name: "a", value: "D:/x", description: "" }])).toBe("D:/x");
    expect(resolveText("${a}", null)).toBe("${a}");
    expect(resolveText("${a}", undefined)).toBe("${a}");
  });

  it("passes text with nothing to resolve straight through", () => {
    expect(resolveText("", { a: "z" })).toBe("");
    expect(resolveText("D:/plain/path", { a: "z" })).toBe("D:/plain/path");
  });
});

describe("variableMap", () => {
  it("coerces non-string values and empty sources", () => {
    expect(variableMap({ a: 3 })).toEqual({ a: "3" });
    expect(variableMap([])).toEqual({});
    expect(variableMap(null)).toEqual({});
  });
});

describe("resolveValue", () => {
  it("touches strings only, at any depth", () => {
    const config = {
      output_dir: "${dir}/q",
      blur_threshold: 80.0,
      flag: true,
      nested: { a: ["${dir}", 3, false, null] },
    };
    expect(resolveValue(config, { dir: "D:/x" })).toEqual({
      output_dir: "D:/x/q",
      blur_threshold: 80.0,
      flag: true,
      nested: { a: ["D:/x", 3, false, null] },
    });
    // And the input is untouched.
    expect(config.output_dir).toBe("${dir}/q");
  });
});

describe("collectRefs", () => {
  it("maps variables to their places, matching the server's field paths", () => {
    const graph: Pick<WorkflowGraph, "nodes"> = {
      nodes: [
        node("n1", "folder", { path: "${dataset_dir}" }),
        node("n2", "prep.quality", {
          output_dir: "${dataset_dir}/rejects",
          iqa_model: "${missing}",
        }),
      ],
    };
    expect(collectRefs(graph)).toEqual({
      dataset_dir: ["n1 · folder.path", "n2 · quality.output_dir"],
      missing: ["n2 · quality.iqa_model"],
    });
  });

  it("walks nested objects and arrays, and dedupes repeats within one field", () => {
    const graph: Pick<WorkflowGraph, "nodes"> = {
      nodes: [
        node("n1", "prep.tag", {
          models: ["${a}", "${a}"],
          overrides: { deep: { deeper: "${b}" } },
          max_tags: 255,
        }),
      ],
    };
    expect(collectRefs(graph)).toEqual({
      a: ["n1 · tag.models[0]", "n1 · tag.models[1]"],
      b: ["n1 · tag.overrides.deep.deeper"],
    });
  });

  it("does not count an escaped $$ as a reference", () => {
    const graph: Pick<WorkflowGraph, "nodes"> = {
      nodes: [node("n1", "tool", { arg: "$$notavar $${a} ${real}" })],
    };
    expect(collectRefs(graph)).toEqual({ real: ["n1 · tool.arg"] });
  });
});

describe("unknownRefs", () => {
  it("reports references the workflow does not define", () => {
    const graph: Pick<WorkflowGraph, "nodes" | "variables"> = {
      variables: [{ name: "dataset_dir", value: "D:/x", description: "" }],
      nodes: [
        node("n1", "folder", { path: "${dataset_dir}" }),
        node("n2", "prep.quality", { output_dir: "${outdir}" }),
      ],
    };
    expect(unknownRefs(graph)).toEqual(["outdir"]);
  });
});

import { describe, expect, it } from "vitest";
import {
  addNode,
  canMove,
  createNode,
  descendantIds,
  defaultNodeConfig,
  legalSources,
  moveNode,
  newNodeId,
  ordinalGlyph,
  ordinals,
  removeNode,
  repointNode,
  sourceChoices,
  type WorkflowGraph,
  type WorkflowNode,
} from "./workflowGraph";
import { defaultCleanForm, defaultCommonForm, defaultTagForm } from "./prepStageConfig";

function node(id: string, type: string, from: string | null = null): WorkflowNode {
  return {
    id,
    type,
    title: id,
    from,
    enabled: true,
    config: {},
    gpu: { required: false, wait: true, device: null },
  };
}

/** folder n1 → tag n2 → quality n3, the shape of the spec's worked example. */
function chain(): WorkflowGraph {
  return {
    version: 1,
    name: "Re-tag character set",
    description: "",
    variables: [],
    nodes: [node("n1", "folder"), node("n2", "prep.tag", "n1"), node("n3", "prep.quality", "n2")],
  };
}

const ids = (graph: WorkflowGraph) => graph.nodes.map((n) => n.id);
const links = (graph: WorkflowGraph) => graph.nodes.map((n) => [n.id, n.from]);

describe("newNodeId", () => {
  it("mints unique random ids, never n<max+1>", () => {
    const minted = new Set(Array.from({ length: 200 }, () => newNodeId()));
    expect(minted.size).toBe(200);
    // Sequential minting would collide across two open tabs and glue saved state to the wrong node.
    expect([...minted].every((id) => !/^n\d+$/.test(id))).toBe(true);
  });
});

describe("createNode", () => {
  it("titles from the catalog and takes the type's GPU default", () => {
    expect(createNode("prep.tag", { id: "x" })).toMatchObject({
      id: "x",
      type: "prep.tag",
      title: "Tag",
      from: null,
      enabled: true,
      gpu: { required: true, wait: true, device: null },
    });
  });

  it("derives the quality GPU default from the config it is given", () => {
    expect(createNode("prep.quality", { config: { metric: "blur" } }).gpu.required).toBe(false);
    expect(createNode("prep.quality", { config: { metric: "iqa" } }).gpu.required).toBe(true);
  });

  it("copies the config instead of aliasing the caller's object", () => {
    const config = { models: ["pixai-v0.9"] };
    const created = createNode("prep.tag", { config });
    created.config.max_tags = 255;
    expect(config).toEqual({ models: ["pixai-v0.9"] });
  });

  /**
   * A node born with `config: {}` runs on the *server's* dataclass defaults — `prep.tag` would
   * tag with two models at `max_tags: 255` while the form showed 40 and the card said "no tagger
   * selected". The step has to carry what the UI shows, opened or not.
   */
  it("is born carrying the form's defaults, never an empty config", () => {
    // `write_mode` is stored once the user picks one; a born node leaves it to the legacy `overwrite`.
    const { write_mode: _unset, ...tagDefaults } = defaultTagForm();
    expect(createNode("prep.tag").config).toEqual({
      ...tagDefaults,
      overrides: {},
    });
    expect(createNode("prep.caption").config).toMatchObject({
      batch_size: 4,
      max_new_tokens: 512,
      prompt_base: "descriptive-long",
    });
    expect(createNode("prep.quality").config).toMatchObject({
      metric: "blur",
      blur_threshold: 80,
      // The form's `move` switch is the server's `action`; the node stores the server's shape.
      action: "report",
    });
    expect(createNode("prep.index").config).toEqual({ models: [] });
    expect(createNode("prep.clean").config).toEqual(defaultCleanForm());
    expect(createNode("folder").config).toEqual(defaultCommonForm());

    for (const type of ["folder", "prep.tag", "prep.caption", "prep.clean", "prep.quality", "prep.index"]) {
      expect(Object.keys(createNode(type).config).length).toBeGreaterThan(0);
    }
  });

  it("lets an explicit config win over the materialized defaults, key by key", () => {
    const created = createNode("prep.tag", { config: { max_tags: 12 } });
    expect(created.config.max_tags).toBe(12);
    // …without dropping the rest of them: a partial config is completed, not taken whole.
    expect(created.config.batch_size).toBe(8);
  });

  it("invents nothing for the two types that have no defaults to invent", () => {
    expect(defaultNodeConfig("train")).toEqual({});
    expect(defaultNodeConfig("unknown.future-type")).toEqual({});
    // A tool node's config is the popover's choice of tool, untouched.
    expect(createNode("tool", { config: { tool_id: "resize", values: {} } }).config).toEqual({
      tool_id: "resize",
      values: {},
    });
  });
});

describe("addNode", () => {
  it("appends and auto-points at the predecessor", () => {
    const graph = addNode(chain(), createNode("prep.caption", { id: "n4" }));
    expect(ids(graph)).toEqual(["n1", "n2", "n3", "n4"]);
    expect(graph.nodes[3].from).toBe("n3");
  });

  it("gives a freshly inserted folder no source — it is a source, not a consumer", () => {
    const graph = addNode(chain(), createNode("folder", { id: "n4" }));
    expect(graph.nodes[3].from).toBeNull();
  });

  it("skips a non-emitting predecessor when auto-pointing", () => {
    const withTrain = addNode(chain(), createNode("train", { id: "n4" }));
    const graph = addNode(withTrain, createNode("prep.caption", { id: "n5" }));
    // n4 (train) is terminal, so the caption node reads n3 instead.
    expect(graph.nodes[4].from).toBe("n3");
  });

  it("inserts in the middle without rewiring anything by default", () => {
    const graph = addNode(chain(), createNode("prep.clean", { id: "nx" }), { at: 2 });
    expect(ids(graph)).toEqual(["n1", "n2", "nx", "n3"]);
    expect(links(graph)).toEqual([
      ["n1", null],
      ["n2", "n1"],
      ["nx", "n2"],
      ["n3", "n2"],
    ]);
  });

  it("splices into the chain on request — the inverse of removeNode", () => {
    const graph = addNode(chain(), createNode("prep.clean", { id: "nx" }), { at: 2, splice: true });
    expect(links(graph)).toEqual([
      ["n1", null],
      ["n2", "n1"],
      ["nx", "n2"],
      ["n3", "nx"],
    ]);
    expect(links(removeNode(graph, "nx"))).toEqual(links(chain()));
  });

  it("never splices a terminal node into the middle of a chain", () => {
    const graph = addNode(chain(), createNode("train", { id: "nt" }), { at: 2, splice: true });
    expect(graph.nodes[3].from).toBe("n2");
  });

  it("clamps an out-of-range index and leaves the input graph untouched", () => {
    const original = chain();
    expect(ids(addNode(original, createNode("tool", { id: "a" }), { at: -5 }))[0]).toBe("a");
    expect(ids(addNode(original, createNode("tool", { id: "b" }), { at: 99 }))[3]).toBe("b");
    expect(ids(original)).toEqual(["n1", "n2", "n3"]);
  });
});

describe("removeNode", () => {
  it("splices the chain: children inherit the deleted node's from", () => {
    expect(links(removeNode(chain(), "n2"))).toEqual([
      ["n1", null],
      ["n3", "n1"],
    ]);
  });

  it("leaves children of a deleted folder sourceless, for validate to report", () => {
    expect(links(removeNode(chain(), "n1"))).toEqual([
      ["n2", null],
      ["n3", "n2"],
    ]);
  });

  it("repoints every child, not just the first", () => {
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [
        node("n1", "folder"),
        node("n2", "prep.tag", "n1"),
        node("n3", "prep.caption", "n2"),
        node("n4", "prep.clean", "n2"),
      ],
    };
    expect(links(removeNode(graph, "n2"))).toEqual([
      ["n1", null],
      ["n3", "n1"],
      ["n4", "n1"],
    ]);
  });

  it("is a no-op for an unknown id, and never mutates the input", () => {
    const original = chain();
    const graph = removeNode(original, "nope");
    expect(ids(graph)).toEqual(["n1", "n2", "n3"]);
    expect(graph).not.toBe(original);
    expect(original.nodes[2].from).toBe("n2");
  });
});

describe("canMove / moveNode", () => {
  it("refuses to lift a node above its own source", () => {
    // n3 reads n2; "Move up" would put it before n2.
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [node("n1", "folder"), node("n2", "prep.tag", "n1"), node("n3", "prep.quality", "n1")],
    };
    expect(canMove(graph, "n3", "up")).toBe(true); // n3 reads n1, not the node above it
    expect(canMove(chain(), "n3", "up")).toBe(false); // n3 reads n2, the node above it
    expect(ids(moveNode(chain(), "n3", "up"))).toEqual(["n1", "n2", "n3"]);
  });

  it("refuses to push a node below the node that reads it", () => {
    expect(canMove(chain(), "n2", "down")).toBe(false);
    expect(ids(moveNode(chain(), "n2", "down"))).toEqual(["n1", "n2", "n3"]);
  });

  it("refuses to move past the ends of the list", () => {
    expect(canMove(chain(), "n1", "up")).toBe(false);
    expect(canMove(chain(), "n3", "down")).toBe(false);
    expect(canMove(chain(), "nope", "up")).toBe(false);
  });

  it("swaps two independent nodes in both directions", () => {
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [node("n1", "folder"), node("n2", "prep.tag", "n1"), node("n3", "prep.quality", "n1")],
    };
    expect(ids(moveNode(graph, "n3", "up"))).toEqual(["n1", "n3", "n2"]);
    expect(ids(moveNode(graph, "n2", "down"))).toEqual(["n1", "n3", "n2"]);
    expect(ids(graph)).toEqual(["n1", "n2", "n3"]);
  });

  it("keeps the backward-only invariant after a legal swap", () => {
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [
        node("n1", "folder"),
        node("n2", "prep.tag", "n1"),
        node("n3", "prep.quality", "n1"),
        node("n4", "train", "n2"),
      ],
    };
    const moved = moveNode(graph, "n3", "up");
    const position = Object.fromEntries(moved.nodes.map((n, i) => [n.id, i]));
    for (const n of moved.nodes) {
      if (n.from) expect(position[n.from]).toBeLessThan(position[n.id]);
    }
  });
});

describe("legalSources", () => {
  it("offers every emitting step, above or below, except itself and its descendants", () => {
    expect(legalSources(chain(), "n1")).toEqual([]); // a folder reads nothing
    expect(legalSources(chain(), "n2").map((n) => n.id)).toEqual(["n1"]); // n3 reads from n2
    expect(legalSources(chain(), "n3").map((n) => n.id)).toEqual(["n1", "n2"]);
    const fan = {
      ...chain(),
      nodes: [node("n1", "folder"), node("n2", "prep.tag", "n1"), node("n3", "prep.quality", "n1")],
    };
    expect(legalSources(fan, "n2").map((n) => n.id)).toEqual(["n1", "n3"]); // n3 is below n2
  });

  it("lists a descendant with the reason it is refused, so no connection fails silently", () => {
    const choices = sourceChoices(chain(), "n2");
    const loop = choices.find((c) => c.node.id === "n3");
    expect(loop).toMatchObject({ allowed: false, below: true });
    expect(loop?.reason).toMatch(/loop/);
    expect(descendantIds(chain(), "n1")).toEqual(new Set(["n2", "n3"]));
  });

  it("skips terminal nodes, which emit nothing", () => {
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [
        node("n1", "folder"),
        node("n2", "train", "n1"),
        node("n3", "prep.tag", "n1"),
      ],
    };
    expect(legalSources(graph, "n3").map((n) => n.id)).toEqual(["n1"]);
  });

  it("offers nothing to a folder, which reads no input", () => {
    const graph = addNode(chain(), createNode("folder", { id: "n4" }));
    expect(legalSources(graph, "n4")).toEqual([]);
    expect(legalSources(graph, "unknown-id")).toEqual([]);
  });
});

describe("repointNode", () => {
  it("accepts an earlier emitting node", () => {
    expect(links(repointNode(chain(), "n3", "n1"))).toEqual([
      ["n1", null],
      ["n2", "n1"],
      ["n3", "n1"],
    ]);
  });

  it("refuses itself, a descendant (a loop) and an unknown id", () => {
    expect(links(repointNode(chain(), "n2", "n3"))).toEqual(links(chain()));
    expect(links(repointNode(chain(), "n2", "n2"))).toEqual(links(chain()));
    expect(links(repointNode(chain(), "n2", "nope"))).toEqual(links(chain()));
  });

  it("clears the source only where null is legal", () => {
    const graph = addNode(chain(), createNode("tool", { id: "n4" }));
    expect(repointNode(graph, "n4", null).nodes[3].from).toBeNull();
    // prep.* must keep a source: a sourceless stage dies mid-run on "needs a dataset 'path'".
    expect(repointNode(graph, "n3", null).nodes[2].from).toBe("n2");
  });
});

describe("repointNode to a step below", () => {
  /** folder n1 -> tag n2 -> quality n3 (reads n2), plus an independent tool n4 and index n5 (reads n4). */
  function wide(): WorkflowGraph {
    return {
      ...chain(),
      nodes: [
        node("n1", "folder"),
        node("n2", "prep.tag", "n1"),
        node("n3", "prep.quality", "n2"),
        node("n4", "tool"),
        node("n5", "prep.index", "n4"),
      ],
    };
  }

  it("moves the step under its new source and keeps the dependents with it", () => {
    const out = repointNode(wide(), "n2", "n4");
    // n2 and its dependent n3 travel together, in their old relative order, right after n4.
    expect(ids(out)).toEqual(["n1", "n4", "n2", "n3", "n5"]);
    expect(out.nodes.find((n) => n.id === "n2")?.from).toBe("n4");
    expect(out.nodes.find((n) => n.id === "n3")?.from).toBe("n2");
  });

  it("always leaves every source above its reader, whichever step is picked", () => {
    for (const node of wide().nodes) {
      for (const choice of sourceChoices(wide(), node.id).filter((c) => c.allowed)) {
        const out = repointNode(wide(), node.id, choice.node.id);
        const position = Object.fromEntries(out.nodes.map((n, i) => [n.id, i]));
        expect(out.nodes).toHaveLength(5);
        for (const n of out.nodes) {
          if (n.from) expect(position[n.from]).toBeLessThan(position[n.id]);
        }
        expect(out.nodes.find((n) => n.id === node.id)?.from).toBe(choice.node.id);
      }
    }
  });

  it("can never create a cycle: every refused source is a descendant", () => {
    const refused = sourceChoices(wide(), "n2").filter((c) => !c.allowed).map((c) => c.node.id);
    expect(refused).toEqual(["n3"]);
    expect(links(repointNode(wide(), "n2", "n3"))).toEqual(links(wide()));
  });

  it("refuses a tool that declares it outputs nothing, with the reason", () => {
    const toolIo = { sink: { input: "folder", output: "none", output_declared: true } } as const;
    const graph = wide();
    graph.nodes[3] = { ...graph.nodes[3], config: { tool_id: "sink" } };
    const choice = sourceChoices(graph, "n2", toolIo).find((c) => c.node.id === "n4");
    expect(choice).toMatchObject({ allowed: false });
    expect(choice?.reason).toMatch(/outputs nothing/);
    expect(links(repointNode(graph, "n2", "n4", toolIo))).toEqual(links(graph));
  });

  it("refuses a source tool whose declared output is not a folder, accepts one that is", () => {
    const graph = wide();
    graph.nodes[3] = { ...graph.nodes[3], config: { tool_id: "extract" } };
    const passthrough = { extract: { input: "none", output: "passthrough", output_declared: true } } as const;
    expect(sourceChoices(graph, "n2", passthrough).find((c) => c.node.id === "n4")?.allowed).toBe(false);
    const folder = { extract: { input: "none", output: "folder", output_declared: true } } as const;
    expect(sourceChoices(graph, "n2", folder).find((c) => c.node.id === "n4")?.allowed).toBe(true);
    // Undeclared: today's behaviour, it may return a folder at run time.
    expect(sourceChoices(graph, "n2", {}).find((c) => c.node.id === "n4")?.allowed).toBe(true);
  });
});

describe("ordinals", () => {
  it("numbers the cards from 1 in list order, following a move", () => {
    const graph: WorkflowGraph = {
      ...chain(),
      nodes: [node("n1", "folder"), node("n2", "prep.tag", "n1"), node("n3", "prep.quality", "n1")],
    };
    expect(ordinals(graph)).toEqual({ n1: 1, n2: 2, n3: 3 });
    expect(ordinals(moveNode(graph, "n3", "up"))).toEqual({ n1: 1, n3: 2, n2: 3 });
  });

  it("renders circled glyphs, falling back to digits past ⑳", () => {
    expect(ordinalGlyph(1)).toBe("①");
    expect(ordinalGlyph(3)).toBe("③");
    expect(ordinalGlyph(20)).toBe("⑳");
    expect(ordinalGlyph(21)).toBe("21");
    expect(ordinalGlyph(0)).toBe("0");
  });
});

describe("createNode with the model registry", () => {
  const registry = [
    { id: "pixai-v0.9", repo_id: "a", downloaded: false, available: true },
    { id: "cl-tagger-1.02", repo_id: "b", downloaded: true, available: true },
    { id: "wd-eva02", repo_id: "c", downloaded: false, available: true },
  ];

  /**
   * A node added and run without opening the drawer used to carry `models: []` / `model: ""` — the
   * pickers' preselect only ran when the form mounted. Seeding happens at birth, by the same rules.
   */
  it("preselects the downloaded taggers, as TagStageForm does", () => {
    expect(createNode("prep.tag", { registry }).config.models).toEqual(["cl-tagger-1.02"]);
  });

  it("falls back to the registry's first two taggers when none is downloaded", () => {
    const none = registry.map((m) => ({ ...m, downloaded: false }));
    expect(createNode("prep.tag", { registry: none }).config.models).toEqual([
      "pixai-v0.9",
      "cl-tagger-1.02",
    ]);
  });

  it("preselects the registry's first caption model, as CaptionStageForm does", () => {
    const captioners = [
      { id: "joycaption-beta-one", repo_id: "x", downloaded: false, available: true },
      { id: "qwen", repo_id: "y", downloaded: true, available: true },
    ];
    expect(createNode("prep.caption", { registry: captioners }).config.model).toBe(
      "joycaption-beta-one"
    );
  });

  it("preselects the registry's first edit-instruction model, as EditCaptionStageForm does", () => {
    const vlms = [
      { id: "qwen3-vl-4b-instruct", repo_id: "x", downloaded: false, available: true },
      { id: "qwen3-vl-8b-instruct", repo_id: "y", downloaded: true, available: true },
    ];
    const born = createNode("prep.edit_caption", { registry: vlms });
    expect(born.config.model).toBe("qwen3-vl-4b-instruct");
    expect(born.config.control_path).toBe("");
    expect(born.gpu.required).toBe(true);
    expect(
      createNode("prep.edit_caption", { registry: vlms, config: { model: "mine" } }).config.model
    ).toBe("mine");
  });

  it("fills a gap, never replaces a choice", () => {
    expect(
      createNode("prep.tag", { registry, config: { models: ["wd-eva02"] } }).config.models
    ).toEqual(["wd-eva02"]);
    expect(
      createNode("prep.caption", { registry, config: { model: "mine" } }).config.model
    ).toBe("mine");
  });

  it("leaves other types and an empty registry alone", () => {
    expect(createNode("prep.index", { registry }).config).toEqual({ models: [] });
    expect(createNode("prep.tag", { registry: [] }).config.models).toEqual(
      createNode("prep.tag").config.models
    );
  });
});

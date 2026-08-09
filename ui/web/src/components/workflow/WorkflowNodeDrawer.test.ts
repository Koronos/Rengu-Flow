/**
 * Smoke coverage for the node drawer, and one real assertion about the part that is easy to get
 * wrong: the Input tab must distinguish a folder the source **saved** from one it is merely
 * **predicted** to emit.
 *
 * Assertions are scoped to a single tab pane on purpose. The drawer's own hint copy contains the
 * words "saved" and "predicted" in other sentences, so a whole-document `toContain` would pass
 * against a component that never rendered the tag at all.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import ElementPlus from "element-plus";
import WorkflowNodeDrawer from "./WorkflowNodeDrawer.vue";
import { api } from "../../api";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCommonForm,
  defaultTagForm,
  type PrepStageForms,
} from "../../lib/prepStageConfig";
import { defaultNodeConfig } from "../../lib/workflowGraph";
import type { PrepModelInfo, PrepPromptOptions } from "../../types/api";
import type { WorkflowGraph, WorkflowNode, WorkflowState } from "../../types/workflow";

vi.mock("../../api", () => ({
  api: {
    workflowNodeLog: vi.fn(async () => ({ chunk: "", offset: 0, progress: null })),
    workflowNodeReport: vi.fn(async () => {
      throw new Error("This step has not written report.json yet.");
    }),
    scanDatasetPath: vi.fn(async () => ({ ok: true, image_count: 12, caption_txt_files: 12 })),
    getSystemStats: vi.fn(async () => ({ ok: true, summary: { gpus: [{ index: 0 }] } })),
    prepModels: vi.fn(async () => ({ models: [] })),
    prepCaptionPrompts: vi.fn(async () => promptCatalogue()),
    prepCaptionPromptPreview: vi.fn(async () => ({ prompt: "", native_format: false })),
    fsStat: vi.fn(async () => ({ exists: true, is_dir: true, is_file: false })),
    cancelWorkflow: vi.fn(async () => ({})),
  },
}));

/** What the registry would preselect if it were allowed to: never what the saved nodes carry. */
function taggerRegistry(): PrepModelInfo[] {
  return [
    { id: "downloaded-A", repo_id: "r/a", downloaded: true, available: true },
    { id: "saved-B", repo_id: "r/b", downloaded: false, available: true },
  ];
}

function captionRegistry(): PrepModelInfo[] {
  return [
    { id: "registry-first", repo_id: "r/first", downloaded: true, available: true },
    { id: "saved-captioner", repo_id: "r/saved", downloaded: true, available: true },
  ];
}

function promptCatalogue(): PrepPromptOptions {
  return {
    bases: [],
    modifiers: [],
    outfit_modes: ["describe", "omit", "mixed"],
    default_base: "registry-base",
    default_modifiers: ["registry-modifier"],
    no_meta: "",
    character_trigger_template: "",
    outfit_texts: {},
    sampling_defaults: {},
  };
}

/** Pane order matches the tab order: Configure, Input, Output, Logs. */
const INPUT_PANE = 1;
const OUTPUT_PANE = 2;

function graphOf(): WorkflowGraph {
  return {
    version: 1,
    name: "Re-tag",
    description: "",
    variables: [{ name: "dataset_dir", value: "D:/datasets/aoi", description: "" }],
    nodes: [
      {
        id: "n1",
        type: "folder",
        title: "Source folder",
        from: null,
        enabled: true,
        config: { path: "${dataset_dir}", caption_format: "sidecar", caption_ext: ".txt" },
        gpu: { required: false, wait: true, device: null },
      },
      {
        id: "n2",
        type: "prep.clean",
        title: "Clean",
        from: "n1",
        enabled: true,
        config: { in_place: false, output_dir: "" },
        gpu: { required: true, wait: true, device: null },
      },
    ],
  };
}

interface MountOptions {
  state?: WorkflowState;
  graph?: WorkflowGraph;
  readOnly?: boolean;
}

async function mountDrawer(nodeId: string, options: MountOptions | WorkflowState = {}) {
  // Back-compat with the three original cases, which pass a bare `state`.
  const opts: MountOptions =
    "state" in options || "graph" in options || "readOnly" in options
      ? (options as MountOptions)
      : { state: options as WorkflowState };
  const graph = opts.graph ?? graphOf();
  const el = document.createElement("div");
  document.body.appendChild(el);
  const errors: unknown[] = [];
  /** Every `update:node` the drawer pushed at its parent — in the app, an autosave each. */
  const updates: WorkflowNode[] = [];
  const app = createApp(WorkflowNodeDrawer, {
    open: true,
    node: graph.nodes.find((n) => n.id === nodeId),
    graph,
    state: opts.state ?? {},
    stale: {},
    workflowId: 7,
    readOnly: opts.readOnly ?? false,
    "onUpdate:node": (node: WorkflowNode) => updates.push(node),
  });
  app.use(ElementPlus);
  app.config.errorHandler = (err) => {
    errors.push(err);
  };
  app.mount(el);
  for (let i = 0; i < 8; i += 1) await nextTick();

  const panes = document.querySelectorAll(".el-tab-pane");
  return {
    app,
    errors,
    updates,
    pane: (index: number) => panes[index]?.textContent ?? "",
  };
}

beforeEach(() => {
  vi.mocked(api.prepModels).mockResolvedValue({ models: [] });
});

afterEach(() => {
  // el-drawer teleports to <body>; without this the next test reads the previous drawer too.
  document.body.innerHTML = "";
});

describe("WorkflowNodeDrawer", () => {
  it("renders the header and the fixed output sentence for a source node", async () => {
    const { app, errors, pane } = await mountDrawer("n1");

    expect(errors).toEqual([]);
    expect(document.body.textContent).toContain("Source folder");
    expect(document.body.textContent).toContain("Not run");
    expect(pane(OUTPUT_PANE)).toContain("Emits its configured folder as the workflow's source");
    // A source has no input to confirm, and says so rather than showing an empty folder row.
    expect(pane(INPUT_PANE)).toContain("it has no input");

    app.unmount();
  });

  it("labels the input predicted, with the variable resolved, while the source has not run", async () => {
    const { app, errors, pane } = await mountDrawer("n2");

    expect(errors).toEqual([]);
    expect(pane(INPUT_PANE)).toContain("D:/datasets/aoi");
    expect(pane(INPUT_PANE)).toContain("predicted");
    expect(pane(INPUT_PANE)).not.toContain("saved");
    // clean(in_place=false) with no output_dir emits <input>/cleaned — the only stage whose
    // output_dir names the result.
    expect(pane(OUTPUT_PANE)).toContain("D:/datasets/aoi/cleaned");

    app.unmount();
  });

  it("labels the same input saved once the source has produced it", async () => {
    const state: WorkflowState = {
      nodes: {
        n1: {
          status: "done",
          output: { path: "D:/datasets/aoi", caption_format: "sidecar", caption_ext: ".txt" },
        },
      },
    };
    const { app, errors, pane } = await mountDrawer("n2", state);

    expect(errors).toEqual([]);
    expect(pane(INPUT_PANE)).toContain("saved");
    expect(pane(INPUT_PANE)).not.toContain("predicted");

    app.unmount();
  });
});

// --------------------------------------------------------------------------- seeding a saved node

/**
 * Opening a step must not *be* an edit.
 *
 * The stage forms fetch their own model registry and preselect from it. That preselect writes
 * into the shared v-model, so if it lands before the node's saved config is seeded back in, the
 * drawer's `builtConfig` watcher sees registry-shaped state, emits `update:node`, and 700 ms
 * later the editor has autosaved it. Workflows keep no history, so that is unrecoverable — which
 * is exactly why these two mounts assert **nothing at all** is pushed at the parent.
 */
const SAVED_TAG_CONFIG = stageSection(
  "tag",
  {
    tagForm: {
      ...defaultTagForm(),
      models: ["saved-B"],
      exclude_tags: ["realistic"],
      max_tags: 55,
      batch_size: 12,
      underscores: true,
    },
    tagThresholds: { "saved-B": { general: 0.4, character: 0.9, rating: 0.6 } },
  },
);

const SAVED_CAPTION_CONFIG = stageSection("caption", {
  captionForm: {
    ...defaultCaptionForm(),
    model: "saved-captioner",
    prompt_base: "saved-base",
    prompt_modifiers: ["saved-modifier"],
    character_name: "Aoi",
    batch_size: 2,
  },
});

/** The `config` a workflow node carries for a stage: the stage section, as the drawer builds it. */
function stageSection(stage: "tag" | "caption", forms: Partial<PrepStageForms>) {
  const payload = buildStageConfig(stage, { form: defaultCommonForm(), ...forms }) as unknown as
    Record<string, Record<string, unknown>>;
  return payload[stage];
}

function prepNode(id: string, type: string, config: Record<string, unknown>): WorkflowNode {
  return {
    id,
    type,
    title: id,
    from: "n1",
    enabled: true,
    config,
    gpu: { required: true, wait: true, device: null },
  };
}

function prepGraph(): WorkflowGraph {
  const base = graphOf();
  return {
    ...base,
    nodes: [
      base.nodes[0],
      prepNode("t1", "prep.tag", SAVED_TAG_CONFIG),
      prepNode("c1", "prep.caption", SAVED_CAPTION_CONFIG),
      prepNode("fresh", "prep.tag", defaultNodeConfig("prep.tag")),
    ],
  };
}

describe("WorkflowNodeDrawer seeding", () => {
  it("leaves a saved prep.tag alone when it is merely opened", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: taggerRegistry() });

    const { app, errors, updates } = await mountDrawer("t1", { graph: prepGraph() });

    expect(errors).toEqual([]);
    // Not one edit: the preselect (`downloaded-A`) must never reach the parent.
    expect(updates.map((node) => node.config)).toEqual([]);

    app.unmount();
  });

  it("leaves a saved prep.caption alone when it is merely opened", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: captionRegistry() });

    const { app, errors, updates } = await mountDrawer("c1", { graph: prepGraph() });

    expect(errors).toEqual([]);
    // The registry's first model, `default_base` and `default_modifiers` would all have landed
    // here — and the one-shot `applySeed` would then have copied the wreckage back.
    expect(updates.map((node) => node.config)).toEqual([]);

    app.unmount();
  });

  it("preselects nothing while the runner owns the workflow", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: taggerRegistry() });

    const { app, updates } = await mountDrawer("fresh", { graph: prepGraph(), readOnly: true });

    // `useWorkflowEditor.mutate` would drop this anyway; not emitting it is what keeps the
    // drawer's picture and the saved graph from diverging behind the user's back.
    expect(updates).toEqual([]);

    app.unmount();
  });

  it("still preselects for a step that has made no choice yet", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: taggerRegistry() });

    const { app, errors, updates } = await mountDrawer("fresh", { graph: prepGraph() });

    expect(errors).toEqual([]);
    // Filling a gap is not overwriting a choice: a node with no models still gets the downloaded
    // ones, and the user sees them before they are saved.
    expect(updates.at(-1)?.config.models).toEqual(["downloaded-A"]);

    app.unmount();
  });
});

// ----------------------------------------------------------------------------------- read-only

/**
 * The drawer is the *only* place stage config, `from`, `gpu` and `enabled` are edited, so a
 * read-only editor that does not reach it leaves every one of those controls looking live while
 * `useWorkflowEditor.mutate` throws the edit away — the keystroke the composable's own contract
 * promises the user will never lose.
 */
describe("WorkflowNodeDrawer read-only", () => {
  /** Element Plus renders a real `<input>` per control, disabled and all. */
  const liveInputs = () =>
    [...document.querySelectorAll<HTMLInputElement>(".node-drawer input")].filter(
      (input) => !input.disabled,
    );

  const runButton = () =>
    [...document.querySelectorAll<HTMLButtonElement>(".node-drawer__title-row button")].find(
      (button) => button.textContent?.trim() === "Run",
    );

  it("leaves every control live while the workflow is idle", async () => {
    const { app } = await mountDrawer("t1", { graph: prepGraph() });

    expect(liveInputs().length).toBeGreaterThan(0);
    expect(runButton()?.disabled).toBe(false);

    app.unmount();
  });

  it("says 'Stop to edit' and disables every control while the runner owns it", async () => {
    const { app, errors } = await mountDrawer("t1", { graph: prepGraph(), readOnly: true });

    expect(errors).toEqual([]);
    // The page's own banner is scrolled out of sight behind the drawer, so it is repeated here.
    expect(document.body.textContent).toContain("Stop to edit");
    expect(document.body.textContent).toContain("editing resumes once it stops");
    // `from`, the GPU switches, `enabled` and every tagging field, in one sweep.
    expect(liveInputs().map((input) => input.outerHTML)).toEqual([]);
    expect(runButton()?.disabled).toBe(true);

    app.unmount();
  });
});

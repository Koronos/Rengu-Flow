/**
 * The nine workflow node types: label, icon, handle behaviour, GPU default and the fixed
 * sentence describing what each one emits.
 *
 * Single source for the add menu, the card summary and the drawer's Output tab — the three
 * places that would otherwise each invent their own wording for the same rule.
 *
 * Mirrors `rengu_flow_ui/workflow_graph.py::NODE_TYPES` and `default_needs_gpu`; the normative
 * table lives in `docs/spec/workflows.md` ("Node catalog"). The trap worth repeating: **`clean`
 * is the only stage whose `output_dir` names the result**. `quality`'s `output_dir` is the
 * *quarantine* folder, so a `quality` node emits its **input** folder — the survivors.
 */

export type WorkflowNodeTypeId =
  | "folder"
  | "prep.tag"
  | "prep.caption"
  | "prep.edit_caption"
  | "prep.clean"
  | "prep.quality"
  | "prep.index"
  | "tool"
  | "train";

/** Add-menu grouping, per the spec's popover: Source · Prepare · Tools · Training. */
export type WorkflowNodeGroupId = "source" | "prepare" | "tools" | "training";

export interface WorkflowNodeTypeSpec {
  type: WorkflowNodeTypeId;
  label: string;
  /** Component name in `@element-plus/icons-vue`; components resolve it, this module stays pure. */
  icon: string;
  group: WorkflowNodeGroupId;
  /** Reads an upstream handle. */
  consumes: boolean;
  /** Produces a handle downstream nodes can read. */
  emits: boolean;
  /** Default for `gpu.required`; see {@link defaultNeedsGpu}. */
  needsGpu: boolean;
  /** May legitimately have `from: null`. */
  sourceOptional: boolean;
}

/** The little a catalog lookup needs from a node — any workflow node satisfies it. */
export interface NodeTypeLike {
  type: string;
  config?: Record<string, unknown> | null;
}

const SPECS: WorkflowNodeTypeSpec[] = [
  {
    type: "folder",
    label: "Source folder",
    icon: "Folder",
    group: "source",
    consumes: false,
    emits: true,
    needsGpu: false,
    sourceOptional: true,
  },
  {
    type: "prep.tag",
    label: "Tag",
    icon: "PriceTag",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "prep.caption",
    label: "Caption",
    icon: "ChatLineSquare",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "prep.edit_caption",
    label: "Edit instructions",
    icon: "EditPen",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "prep.clean",
    label: "Clean",
    icon: "Scissor",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "prep.quality",
    label: "Quality filter",
    icon: "Filter",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "prep.index",
    label: "Quality index",
    icon: "DataAnalysis",
    group: "prepare",
    consumes: true,
    emits: true,
    needsGpu: true,
    sourceOptional: false,
  },
  {
    type: "tool",
    label: "Tool",
    icon: "Tools",
    group: "tools",
    consumes: true,
    emits: true,
    needsGpu: false,
    sourceOptional: true,
  },
  {
    type: "train",
    label: "Training run",
    icon: "VideoPlay",
    group: "training",
    consumes: true,
    emits: false,
    needsGpu: false,
    sourceOptional: false,
  },
];

export const NODE_TYPE_LIST: readonly WorkflowNodeTypeSpec[] = SPECS;

export const NODE_TYPES: Record<WorkflowNodeTypeId, WorkflowNodeTypeSpec> = SPECS.reduce(
  (acc, spec) => {
    acc[spec.type] = spec;
    return acc;
  },
  {} as Record<WorkflowNodeTypeId, WorkflowNodeTypeSpec>,
);

export interface WorkflowNodeGroup {
  id: WorkflowNodeGroupId;
  label: string;
  types: readonly WorkflowNodeTypeSpec[];
}

/** The add popover, in menu order. `tools` is seeded from the Toolbox list at render time. */
export const NODE_TYPE_GROUPS: readonly WorkflowNodeGroup[] = (
  [
    ["source", "Source"],
    ["prepare", "Prepare"],
    ["tools", "Tools"],
    ["training", "Training"],
  ] as const
).map(([id, label]) => ({ id, label, types: SPECS.filter((spec) => spec.group === id) }));

export function nodeTypeSpec(type: string): WorkflowNodeTypeSpec | undefined {
  return NODE_TYPES[type as WorkflowNodeTypeId];
}

/** A human label, falling back to the raw type so a graph from a newer app still reads. */
export function nodeTypeLabel(type: string): string {
  return nodeTypeSpec(type)?.label ?? type;
}

export function nodeTypeIcon(type: string): string {
  return nodeTypeSpec(type)?.icon ?? "QuestionFilled";
}

/**
 * Unknown types are treated as consuming and emitting so that downgrading the app never quietly
 * drops the links around a node it does not understand.
 */
export function consumesInput(type: string): boolean {
  return nodeTypeSpec(type)?.consumes ?? true;
}

export function emitsHandle(type: string): boolean {
  return nodeTypeSpec(type)?.emits ?? true;
}

/** Whether `from: null` is legal. Unknown types are not judged — parity with `validate`. */
export function sourceMayBeEmpty(type: string): boolean {
  return nodeTypeSpec(type)?.sourceOptional ?? true;
}

/**
 * Default for `gpu.required`; the user can always override it on the node.
 *
 * `prep.quality` follows its metric: `blur` is a pure-CPU Laplacian variance with no extra
 * dependencies, while `aesthetic` and `iqa` load models.
 */
export function defaultNeedsGpu(type: string, config?: Record<string, unknown> | null): boolean {
  const spec = nodeTypeSpec(type);
  if (!spec) return false;
  if (type === "prep.quality") return (config?.metric ?? "blur") !== "blur";
  return spec.needsGpu;
}

/** What a Toolbox tool declares about the folder it takes and gives (`io` in its `tool.json`). */
export interface ToolIo {
  input: "folder" | "none";
  output: "folder" | "passthrough" | "none";
  /** `true` when the author declared the output; a workflow only judges what was declared. */
  output_declared?: boolean;
  input_declared?: boolean;
}

/** `tool_id` -> its declared io, as the editor loads it from the Toolbox list. */
export type ToolIoMap = Record<string, ToolIo | undefined>;

/** The io of the tool behind a `tool` node; `undefined` for any other node or an unknown tool. */
export function toolIoOf(node: NodeTypeLike, map: ToolIoMap | undefined): ToolIo | undefined {
  if (node.type !== "tool") return undefined;
  const id = node.config?.tool_id;
  return typeof id === "string" && id ? map?.[id] : undefined;
}

/** What a tool takes, in the words the From select and the tool form share. */
export function describeToolInput(io: ToolIo | undefined): string {
  if (!io) return "";
  return io.input === "folder"
    ? "Takes the incoming dataset folder"
    : "Takes no folder (it can start a workflow)";
}

/** What a tool gives, in the words the From select and the tool form share. */
export function describeToolOutput(io: ToolIo | undefined): string {
  if (!io || io.output === "passthrough") {
    return "Works in place: passes its input folder on (or the folder it returns)";
  }
  if (io.output === "folder") return "Returns a new folder, decided when it runs";
  return "Hands nothing on: no step can read from it";
}

/** The caption layout a tag / caption / edit-caption step writes, as a short phrase (or `""`). */
export function outputFormatChoice(
  config: Record<string, unknown> | null | undefined,
): "inherit" | "sidecar" | "json" {
  // Twin of `workflow_graph.output_format_choice`: trimmed, lower-cased, unknown = inherit.
  const value = String(config?.output_format ?? "").trim().toLowerCase();
  return value === "json" || value === "sidecar" ? value : "inherit";
}

export function outputFormatLabel(config: Record<string, unknown> | null | undefined): string {
  const format = outputFormatChoice(config);
  if (format === "json") return "captions.json";
  if (format === "sidecar") {
    const ext = typeof config?.output_ext === "string" && config.output_ext ? config.output_ext : ".txt";
    return `sidecar files (${ext.startsWith(".") ? ext : `.${ext}`})`;
  }
  return "";
}

/**
 * The fixed sentence for a type's output rule — what the card summary and the drawer's Output
 * tab both show. Only `prep.clean` varies with config, because its rule genuinely does
 * (`in_place ? input : (output_dir or <input>/cleaned)`); the caption steps add the layout they
 * convert to, and a tool follows what it declares.
 */
export function describeOutput(node: NodeTypeLike, toolIo?: ToolIo): string {
  const config = node.config ?? {};
  const converts = outputFormatLabel(config);
  switch (node.type) {
    case "folder":
      return "Emits its configured folder as the workflow's source";
    case "prep.tag":
      return converts
        ? `Writes tags into the input folder as ${converts} and emits it`
        : "Writes tag sidecars into the input folder and emits it unchanged";
    case "prep.caption":
      return converts
        ? `Writes captions into the input folder as ${converts} and emits it`
        : "Writes captions into the input folder and emits it unchanged";
    case "prep.edit_caption":
      return converts
        ? `Writes edit instructions into each target's caption as ${converts} and emits the input folder`
        : "Writes edit instructions into line 1 of each target's caption and emits the input folder unchanged";
    case "prep.clean": {
      if (config.in_place) return "Emits the input folder; images are cleaned in place";
      const outputDir = typeof config.output_dir === "string" ? config.output_dir.trim() : "";
      return outputDir ? `Emits ${outputDir}` : "Emits <input>/cleaned";
    }
    case "prep.quality":
      return "Emits the input folder; flagged images are moved to the quarantine folder";
    case "prep.index":
      return "Writes the quality index outside the dataset and emits the input folder unchanged";
    case "tool":
      if (toolIo?.output === "folder") return "Emits the new folder the tool returns";
      if (toolIo?.output === "none") return "Emits nothing; no step can read from this tool";
      return "Emits the folder the tool returns, or the input folder unchanged when it returns nothing";
    case "train":
      return "Emits nothing; training is the end of the chain";
    default:
      return `Unknown node type ${node.type}; this step cannot run in this version`;
  }
}

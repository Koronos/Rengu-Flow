/**
 * Pure edit operations on a workflow graph. Every one returns a **new** graph; nothing is
 * mutated in place, so the editor can diff, undo and stay reactive without deep watchers.
 *
 * The invariant that makes all of this simple: **`from` always points at an EARLIER node.**
 * Cycles are then impossible to express, so there is no cycle detection and no topological sort
 * anywhere — list order is execution order. The user may *pick* any step as a source (even one
 * below), because {@link repointNode} then **moves the step — with the steps that read from it —
 * to right after that source**, so the invariant holds again; the one source it refuses is a
 * descendant (that would be a loop). {@link canMove} refuses to lift a node above its own source,
 * and {@link removeNode} splices children onto the deleted node's `from`.
 *
 * Note there is deliberately **no `workflowHash`**: staleness is computed server-side only.
 * `JSON.stringify` emits `80` where `json.dumps` emits `80.0`, and `QualityStageConfig`'s
 * `blur_threshold` default is `80.0`, so a client recomputation would disagree on every
 * `prep.quality` node. The client renders the `stale` flag from the state payload.
 *
 * Shapes mirror `rengu_flow_ui/workflow_graph.py` (whose `source` is this JSON's `from`).
 */

import { preselectModel, preselectTagModels } from "./modelPreselect";
import { buildStageConfig, defaultCommonForm } from "./prepStageConfig";
import {
  consumesInput,
  defaultNeedsGpu,
  emitsHandle,
  nodeTypeLabel,
  sourceMayBeEmpty,
  toolIoOf,
  type ToolIoMap,
} from "./workflowNodeTypes";
import type { PrepModelInfo, PrepStage } from "@/types/api";

// The graph shapes live in `types/workflow.ts`, next to the API payloads they travel in.
// Re-exported here so callers of the editing helpers get them from one import.
export type {
  WorkflowGraph,
  WorkflowNode,
  WorkflowNodeGpu,
  WorkflowVariable,
} from "@/types/workflow";

import type {
  WorkflowGraph,
  WorkflowNode,
  WorkflowNodeGpu,
  WorkflowVariable,
} from "@/types/workflow";

export type MoveDirection = "up" | "down";

export interface CreateNodeOptions {
  id?: string;
  title?: string;
  config?: Record<string, unknown>;
  gpu?: Partial<WorkflowNodeGpu>;
  enabled?: boolean;
  /**
   * The stage's model registry (`GET /prep/models?stage=`), when the caller has it. Seeds the
   * model picker the way the stage form's preselect would — see {@link seedModelDefaults}.
   */
  registry?: readonly PrepModelInfo[];
}

export interface AddNodeOptions {
  /** Insertion index; anything out of range (or omitted) appends. */
  at?: number;
  /**
   * Re-point the nodes that read the new node's predecessor so they read the new node instead —
   * the exact inverse of {@link removeNode}'s splice. Off by default: inserting a step should
   * not silently rewire a chain the user did not touch. The "insert here" affordance turns it on.
   */
  splice?: boolean;
}

/**
 * Ids are opaque, stable and **random** — never `n<max+1>`. With two tabs open, sequential
 * minting hands the same `n5` to two different node types; the second save wins and inherits a
 * `state_json["n5"]` written by a node of another stage, so the card shows someone else's run.
 */
export function newNodeId(): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  if (uuid) return uuid;
  return `n-${Math.random().toString(36).slice(2, 10)}`;
}

/** Node type -> the prep stage whose form owns its config. Mirrors the drawer's own table. */
const PREP_STAGES: Record<string, PrepStage> = {
  "prep.tag": "tag",
  "prep.caption": "caption",
  "prep.edit_caption": "edit_caption",
  "prep.clean": "clean",
  "prep.quality": "quality",
  "prep.index": "index",
};

/**
 * The config a node is **born** with: this app's form defaults, materialized.
 *
 * A node created with `config: {}` runs on the *server's* dataclass defaults, which are not the
 * ones the UI shows — `prep.tag` would run `pixai-v0.9` + `cl-tagger-1.02` at `max_tags: 255`
 * while its card printed "no tagger selected" and its form showed 40. Writing the form's own
 * defaults in at creation keeps the promise the editor makes everywhere else: **what the user
 * sees is what runs**, whether or not they ever opened the step.
 *
 * The values come from {@link buildStageConfig}'s own `default*Form()` fallbacks, so there is
 * exactly one copy of them; `folder`'s config *is* a dataset handle, so it takes the common form.
 * Everything else (`tool`, `train`) has no defaults worth inventing — the popover supplies a
 * tool's `tool_id`, and a `train` node is a picker.
 */
export function defaultNodeConfig(type: string): Record<string, unknown> {
  if (type === "folder") return { ...defaultCommonForm() };
  const stage = PREP_STAGES[type];
  if (!stage) return {};
  const payload = buildStageConfig(stage, { form: defaultCommonForm() }) as unknown as Record<
    string,
    unknown
  >;
  return { ...(payload[stage] as Record<string, unknown>) };
}

/**
 * Fill an empty model picker from the registry, by the stage forms' own preselect rules
 * (`modelPreselect.ts`, which the forms read too): tag takes the downloaded taggers, else the
 * registry's first two; caption and edit_caption take the registry's first model. A choice
 * already made is never replaced.
 *
 * Without this a step added and run without opening its drawer carried `models: []` /
 * `model: ""` — the preselect lived only in the form's mount — and the server refused it (tag) or
 * ran it with no model (caption).
 */
export function seedModelDefaults(
  type: string,
  config: Record<string, unknown>,
  registry: readonly PrepModelInfo[] | undefined,
): Record<string, unknown> {
  if (!registry?.length) return config;
  if (type === "prep.tag") {
    const current = config.models;
    if (Array.isArray(current) && current.length) return config;
    return { ...config, models: preselectTagModels(registry) };
  }
  if (type === "prep.caption" || type === "prep.edit_caption") {
    if (typeof config.model === "string" && config.model) return config;
    return { ...config, model: preselectModel(registry) };
  }
  return config;
}

/** A node with this app's defaults filled in. Its `from` is decided by {@link addNode}. */
export function createNode(type: string, options: CreateNodeOptions = {}): WorkflowNode {
  const config = seedModelDefaults(
    type,
    { ...defaultNodeConfig(type), ...(options.config ?? {}) },
    options.registry,
  );
  return {
    id: options.id ?? newNodeId(),
    type,
    title: options.title ?? nodeTypeLabel(type),
    from: null,
    enabled: options.enabled ?? true,
    config,
    gpu: {
      required: defaultNeedsGpu(type, config),
      wait: true,
      device: null,
      ...(options.gpu ?? {}),
    },
  };
}

function cloneGraph(graph: WorkflowGraph, nodes: WorkflowNode[]): WorkflowGraph {
  return { ...graph, nodes };
}

function indexOfNode(graph: WorkflowGraph, id: string): number {
  return graph.nodes.findIndex((node) => node.id === id);
}

/** The nearest preceding node that emits a handle — what a new consumer should read. */
function lastEmittingBefore(nodes: readonly WorkflowNode[], index: number): string | null {
  for (let i = Math.min(index, nodes.length) - 1; i >= 0; i -= 1) {
    if (emitsHandle(nodes[i].type)) return nodes[i].id;
  }
  return null;
}

/**
 * Insert *node* at `options.at` (default: the end).
 *
 * A new node's `from` auto-points at its nearest emitting predecessor — **except a `folder`**,
 * which gets `from: null`. A folder is a source, not a consumer; auto-pointing it would create a
 * link the executor ignores and the reader misreads.
 */
export function addNode(
  graph: WorkflowGraph,
  node: WorkflowNode,
  options: AddNodeOptions = {},
): WorkflowGraph {
  const nodes = [...graph.nodes];
  const requested = options.at;
  const at =
    typeof requested === "number" && Number.isFinite(requested)
      ? Math.max(0, Math.min(Math.trunc(requested), nodes.length))
      : nodes.length;

  const predecessor = at > 0 ? nodes[at - 1] : undefined;
  const inserted: WorkflowNode = {
    ...node,
    from: consumesInput(node.type) ? lastEmittingBefore(nodes, at) : null,
  };

  if (options.splice && predecessor && consumesInput(node.type) && emitsHandle(node.type)) {
    for (let i = at; i < nodes.length; i += 1) {
      if (nodes[i].from === predecessor.id) nodes[i] = { ...nodes[i], from: inserted.id };
    }
  }
  nodes.splice(at, 0, inserted);
  return cloneGraph(graph, nodes);
}

/**
 * Remove a node and **splice the chain**: its children inherit its `from`. That is the least
 * surprising repair — the alternative, orphaning them, breaks pre-flight for every node below.
 *
 * Deleting a `folder` therefore leaves its children with `from: null`, which `validate` reports
 * ("③ has no source. Pick a new source folder first.") rather than letting it die mid-run.
 */
export function removeNode(graph: WorkflowGraph, id: string): WorkflowGraph {
  const target = graph.nodes.find((node) => node.id === id);
  if (!target) return cloneGraph(graph, [...graph.nodes]);
  const nodes = graph.nodes
    .filter((node) => node.id !== id)
    .map((node) => (node.from === id ? { ...node, from: target.from } : node));
  return cloneGraph(graph, nodes);
}

/**
 * Whether `⋮ → Move up / Move down` is available.
 *
 * Blocked when the swap would put a node before its own source — in either direction, since
 * moving A down past B is the same edit as moving B up past A.
 */
export function canMove(graph: WorkflowGraph, id: string, direction: MoveDirection): boolean {
  const index = indexOfNode(graph, id);
  if (index < 0) return false;
  const node = graph.nodes[index];
  if (direction === "up") {
    if (index === 0) return false;
    return node.from !== graph.nodes[index - 1].id;
  }
  if (index >= graph.nodes.length - 1) return false;
  return graph.nodes[index + 1].from !== node.id;
}

/** Swap a node with its neighbour. An illegal move is a no-op, not a thrown error. */
export function moveNode(graph: WorkflowGraph, id: string, direction: MoveDirection): WorkflowGraph {
  const nodes = [...graph.nodes];
  if (!canMove(graph, id, direction)) return cloneGraph(graph, nodes);
  const index = indexOfNode(graph, id);
  const target = direction === "up" ? index - 1 : index + 1;
  [nodes[index], nodes[target]] = [nodes[target], nodes[index]];
  return cloneGraph(graph, nodes);
}

/** Every step that (transitively) reads from `id` — the ones a source for `id` must not be. */
export function descendantIds(graph: WorkflowGraph, id: string): Set<string> {
  const out = new Set<string>();
  const queue = [id];
  while (queue.length) {
    const current = queue.pop() as string;
    for (const node of graph.nodes) {
      if (node.from === current && !out.has(node.id)) {
        out.add(node.id);
        queue.push(node.id);
      }
    }
  }
  return out;
}

/**
 * Why a tool cannot hand a folder to the step reading it, or `""` when it can — the client twin of
 * `workflow_graph._source_problem`. Only tools can fail to: a tool declaring `output: none`, or a
 * *source* tool (no input) whose declared output is not a folder. A pass-through tool is looked
 * through to what feeds it; a tool that declares nothing is left alone (it may return a folder).
 */
export function toolSourceProblem(
  graph: WorkflowGraph,
  source: WorkflowNode,
  toolIo?: ToolIoMap,
): string {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const seen = new Set<string>();
  let current: WorkflowNode | undefined = source;
  while (current && current.type === "tool" && current.enabled) {
    if (seen.has(current.id)) return "";
    seen.add(current.id);
    const io = toolIoOf(current, toolIo);
    const output = io?.output ?? "passthrough";
    if (output === "none") return "declares that it outputs nothing, so there is no folder to read";
    if (output === "folder") return "";
    if (!current.from) {
      return io?.output_declared
        ? "is a source (it has no input) and does not declare a folder output; set its output to 'folder' in the Toolbox"
        : "";
    }
    current = byId.get(current.from);
  }
  return "";
}

/** One row of the From select: a step, and whether (and why not) it can feed `nodeId`. */
export interface SourceChoice {
  node: WorkflowNode;
  allowed: boolean;
  /** Why it is not offered as a source; `""` when it is. */
  reason: string;
  /** The step sits below `nodeId`: picking it moves `nodeId` (and what reads from it) under it. */
  below: boolean;
}

/**
 * Every step that emits a folder, with whether it may be `nodeId`'s source.
 *
 * Any step may — above or below — except `nodeId` itself, the steps that read from `nodeId`
 * (directly or not: that would be a loop) and a tool that hands nothing on. The disallowed ones
 * stay in the list with their reason, so a step that cannot be connected says why instead of
 * silently not being there. A node that does not consume (`folder`) has no choices at all.
 */
export function sourceChoices(
  graph: WorkflowGraph,
  nodeId: string,
  toolIo?: ToolIoMap,
): SourceChoice[] {
  const index = indexOfNode(graph, nodeId);
  if (index < 0 || !consumesInput(graph.nodes[index].type)) return [];
  const descendants = descendantIds(graph, nodeId);
  const choices: SourceChoice[] = [];
  graph.nodes.forEach((node, position) => {
    if (node.id === nodeId || !emitsHandle(node.type)) return;
    let reason = "";
    if (descendants.has(node.id)) reason = "it reads from this step, so connecting it would make a loop";
    else if (node.type === "tool") {
      const problem = toolSourceProblem(graph, node, toolIo);
      if (problem) reason = `it ${problem}`;
    }
    choices.push({ node, allowed: !reason, reason, below: position > index });
  });
  return choices;
}

/**
 * The nodes a given node may read from: every emitting step except itself and its descendants
 * (see {@link sourceChoices} for the ones refused, with reasons). `train` never appears — it is
 * terminal and emits nothing.
 */
export function legalSources(
  graph: WorkflowGraph,
  nodeId: string,
  toolIo?: ToolIoMap,
): WorkflowNode[] {
  return sourceChoices(graph, nodeId, toolIo)
    .filter((choice) => choice.allowed)
    .map((choice) => choice.node);
}

/**
 * Re-point a node's `from`. An illegal source (a loop, itself, an unknown id, a tool that hands
 * nothing on) is a no-op — callers that want to say why use {@link sourceChoices}.
 *
 * A source **below** the node is allowed: the node moves, together with the steps that read from
 * it (keeping their relative order), to right after that source, so "a source is always above its
 * reader" — and with it list order == execution order — still holds. Everything else keeps its
 * place.
 */
export function repointNode(
  graph: WorkflowGraph,
  id: string,
  sourceId: string | null,
  toolIo?: ToolIoMap,
): WorkflowGraph {
  const index = indexOfNode(graph, id);
  const nodes = [...graph.nodes];
  if (index < 0) return cloneGraph(graph, nodes);
  const node = nodes[index];

  if (sourceId === null) {
    if (!sourceMayBeEmpty(node.type)) return cloneGraph(graph, nodes);
    nodes[index] = { ...node, from: null };
    return cloneGraph(graph, nodes);
  }
  if (!legalSources(graph, id, toolIo).some((candidate) => candidate.id === sourceId)) {
    return cloneGraph(graph, nodes);
  }

  const sourceIndex = indexOfNode(graph, sourceId);
  if (sourceIndex < index) {
    nodes[index] = { ...node, from: sourceId };
    return cloneGraph(graph, nodes);
  }

  const block = new Set([id, ...descendantIds(graph, id)]);
  const moved = graph.nodes
    .filter((candidate) => block.has(candidate.id))
    .map((candidate) => (candidate.id === id ? { ...candidate, from: sourceId } : candidate));
  const rest = graph.nodes.filter((candidate) => !block.has(candidate.id));
  const after = rest.findIndex((candidate) => candidate.id === sourceId) + 1;
  rest.splice(after, 0, ...moved);
  return cloneGraph(graph, rest);
}

/** Node id -> its 1-based position, the ① ② ③ the cards and the `⟵ from ①` badge show. */
export function ordinals(graph: WorkflowGraph): Record<string, number> {
  const out: Record<string, number> = {};
  graph.nodes.forEach((node, index) => {
    out[node.id] = index + 1;
  });
  return out;
}

const ORDINAL_GLYPHS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳";

/** The circled digit for a 1-based position; plain digits past the glyphs Unicode provides. */
export function ordinalGlyph(position: number): string {
  if (!Number.isInteger(position) || position < 1 || position > ORDINAL_GLYPHS.length) {
    return String(position);
  }
  return ORDINAL_GLYPHS[position - 1];
}

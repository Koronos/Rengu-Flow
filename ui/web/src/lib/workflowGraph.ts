/**
 * Pure edit operations on a workflow graph. Every one returns a **new** graph; nothing is
 * mutated in place, so the editor can diff, undo and stay reactive without deep watchers.
 *
 * The invariant that makes all of this simple: **`from` may only point at an EARLIER node.**
 * Cycles are then impossible to express, so there is no cycle detection and no topological sort
 * anywhere — list order is execution order. Every operation here preserves it:
 * {@link legalSources} only offers precedents, {@link canMove} refuses to lift a node above its
 * own source, and {@link removeNode} splices children onto the deleted node's `from`.
 *
 * Note there is deliberately **no `workflowHash`**: staleness is computed server-side only.
 * `JSON.stringify` emits `80` where `json.dumps` emits `80.0`, and `QualityStageConfig`'s
 * `blur_threshold` default is `80.0`, so a client recomputation would disagree on every
 * `prep.quality` node. The client renders the `stale` flag from the state payload.
 *
 * Shapes mirror `rengu_flow_ui/workflow_graph.py` (whose `source` is this JSON's `from`).
 */

import { buildStageConfig, defaultCommonForm } from "./prepStageConfig";
import {
  consumesInput,
  defaultNeedsGpu,
  emitsHandle,
  nodeTypeLabel,
  sourceMayBeEmpty,
} from "./workflowNodeTypes";
import type { PrepStage } from "@/types/api";

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

/** A node with this app's defaults filled in. Its `from` is decided by {@link addNode}. */
export function createNode(type: string, options: CreateNodeOptions = {}): WorkflowNode {
  const config = { ...defaultNodeConfig(type), ...(options.config ?? {}) };
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

/**
 * The nodes a given node may read from: **strictly earlier** ones that emit a handle.
 *
 * A node that does not consume (`folder`) has none, and `train` never appears in the list — it
 * is terminal and emits nothing.
 */
export function legalSources(graph: WorkflowGraph, nodeId: string): WorkflowNode[] {
  const index = indexOfNode(graph, nodeId);
  if (index < 0) return [];
  if (!consumesInput(graph.nodes[index].type)) return [];
  return graph.nodes.slice(0, index).filter((node) => emitsHandle(node.type));
}

/** Re-point a node's `from`. An illegal source (or a forward one) is a no-op. */
export function repointNode(
  graph: WorkflowGraph,
  id: string,
  sourceId: string | null,
): WorkflowGraph {
  const index = indexOfNode(graph, id);
  const nodes = [...graph.nodes];
  if (index < 0) return cloneGraph(graph, nodes);
  const node = nodes[index];

  if (sourceId === null) {
    if (!sourceMayBeEmpty(node.type)) return cloneGraph(graph, nodes);
  } else if (!legalSources(graph, id).some((candidate) => candidate.id === sourceId)) {
    return cloneGraph(graph, nodes);
  }
  nodes[index] = { ...node, from: sourceId };
  return cloneGraph(graph, nodes);
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

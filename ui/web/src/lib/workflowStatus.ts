/**
 * What a node card and the run bar *say* — status chips, the set `[ > Run ]` will execute, and the
 * two "you cannot do that yet, here is the repair" messages.
 *
 * All of it is pure and derived from the server's payload. In particular **`stale` is never
 * recomputed here**: it arrives in `WorkflowDetail.stale` because the hash is server-side
 * (`JSON.stringify` emits `80` where `json.dumps` emits `80.0`, and `QualityStageConfig`'s
 * `blur_threshold` default is exactly `80.0`). This module only decides how to *paint* it — and a
 * node that is `done` **and** `stale` is painted as both, because that combination is the
 * information the user needs, not a contradiction to resolve.
 */

import { canMove, ordinalGlyph, ordinals, type MoveDirection } from "./workflowGraph";
import type {
  NodeState,
  NodeStatus,
  WorkflowGraph,
  WorkflowNode,
  WorkflowState,
  WorkflowStatus,
} from "../types/workflow";

export type ChipTone = "info" | "warning" | "primary" | "success" | "danger";

export interface NodeChip {
  /** The status glyph from the spec's table: one character, never an icon font. */
  glyph: string;
  label: string;
  tone: ChipTone;
  /** Second line: the GPU wait reason, the first line of an error, the queued run. */
  detail: string;
  /** Whether the card renders the slim progress bar under the chip. */
  showProgress: boolean;
}

/** The workflow statuses that make the editor read-only ("Stop to edit"). */
export function isBusy(status: WorkflowStatus | undefined): boolean {
  return status === "running" || status === "cancelling";
}

/** One node's saved entry, or `undefined` when it has never run. */
export function nodeEntry(
  state: WorkflowState | null | undefined,
  nodeId: string
): NodeState | undefined {
  return state?.nodes?.[nodeId];
}

/** The first non-empty line of a multi-line error — all a chip has room for. */
export function firstLine(text: unknown): string {
  if (typeof text !== "string") return "";
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

/** `2026-08-09T12:34:56.789+00:00` -> `12:34`; anything unparseable comes back untouched. */
export function clockTime(iso: unknown): string {
  if (typeof iso !== "string" || !iso) return "";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "";
  return `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

function trainDetail(entry: NodeState | undefined): string {
  const result = entry?.result as { job_id?: unknown } | null | undefined;
  const jobId = result?.job_id;
  return jobId == null ? "" : `Queued run #${jobId}`;
}

/**
 * The chip for one node.
 *
 * `pending` is deliberately two different chips: under a running workflow it is *queued*, and
 * under an idle one it is *not run*. Same wire value, opposite meaning to the reader.
 */
export function nodeChip(
  node: WorkflowNode,
  entry: NodeState | undefined,
  workflowStatus?: WorkflowStatus
): NodeChip {
  if (!node.enabled) {
    return { glyph: "—", label: "Disabled", tone: "info", detail: "", showProgress: false };
  }

  const status: NodeStatus = entry?.status ?? "pending";
  const error = firstLine(entry?.error);

  switch (status) {
    case "waiting_gpu":
      return {
        glyph: "◷",
        label: "Waiting for GPU",
        tone: "warning",
        detail: error || "Waiting for the GPU.",
        showProgress: false,
      };
    case "launching":
      return {
        glyph: "◷",
        label: "Launching",
        tone: "primary",
        // Parked by the request that started it: the poller installs the extras, then spawns.
        detail: entry?.install_pending ? "Installing the prep dependencies first." : "",
        showProgress: true,
      };
    case "running":
      return { glyph: "⟳", label: "Running", tone: "primary", detail: "", showProgress: true };
    case "stopping":
      return { glyph: "⟳", label: "Stopping", tone: "warning", detail: "", showProgress: true };
    case "done": {
      const finished = clockTime(entry?.finished_at);
      return {
        glyph: "✓",
        label: "Done",
        tone: "success",
        detail: trainDetail(entry) || (finished ? `Finished ${finished}` : ""),
        showProgress: false,
      };
    }
    case "failed":
      return {
        glyph: "✕",
        label: "Failed",
        tone: "danger",
        detail: error || "The step failed; open Logs for the traceback.",
        showProgress: false,
      };
    case "stopped":
      return {
        glyph: "■",
        label: "Stopped",
        tone: "warning",
        // Prep stages resume; saying so is what stops a user re-running from the top.
        detail: "Stopped part-way; running again resumes it.",
        showProgress: false,
      };
    case "skipped":
      return { glyph: "—", label: "Skipped", tone: "info", detail: "", showProgress: false };
    default:
      return isBusy(workflowStatus)
        ? { glyph: "◷", label: "Queued", tone: "info", detail: "", showProgress: false }
        : { glyph: "●", label: "Not run", tone: "info", detail: "", showProgress: false };
  }
}

/**
 * Whether `[ > Run ]` would execute this node: **everything enabled that is not done-and-fresh**,
 * i.e. idle + stale + failed + stopped.
 *
 * `stopped` belongs in the set and the point is not academic: a caption stage stopped at 60 % that
 * `Run` skipped would leave the remaining 40 % uncaptioned and train on it.
 */
export function shouldRunNode(
  node: WorkflowNode,
  entry: NodeState | undefined,
  stale: boolean
): boolean {
  if (!node.enabled) return false;
  if (entry?.status === "done") return stale;
  return true;
}

/** The ids `[ > Run ]` would execute, in list order. */
export function nodesToRun(
  graph: WorkflowGraph,
  state: WorkflowState | null | undefined,
  stale: Record<string, boolean> | null | undefined
): string[] {
  return graph.nodes
    .filter((node) => shouldRunNode(node, nodeEntry(state, node.id), Boolean(stale?.[node.id])))
    .map((node) => node.id);
}

function hasSavedOutput(state: WorkflowState | null | undefined, nodeId: string): boolean {
  return Boolean(nodeEntry(state, nodeId)?.output);
}

/**
 * Why *Run from here* is unavailable, or `""` when it is.
 *
 * Starting mid-chain reuses the upstream nodes' **saved** handles; without one there is nothing to
 * feed the chain and the run would die on the first stage's "Prep config needs a dataset 'path'".
 * The whole ancestor chain is checked, not just the direct parent — the server's
 * `workflow_runner._require_saved_ancestors` does, and a menu entry it will refuse is a lie. The
 * message names the **earliest** gap, exactly as the server's does.
 */
export function runFromBlockReason(
  graph: WorkflowGraph,
  nodeId: string,
  state: WorkflowState | null | undefined
): string {
  const byId = new Map(graph.nodes.map((candidate) => [candidate.id, candidate]));
  const node = byId.get(nodeId);
  if (!node) return "";

  const positions = ordinals(graph);
  const missing: number[] = [];
  const seen = new Set<string>();
  for (let cursor = node.from; cursor && !seen.has(cursor); cursor = byId.get(cursor)?.from ?? null) {
    seen.add(cursor);
    if (!byId.has(cursor)) break;
    if (!hasSavedOutput(state, cursor)) missing.push(positions[cursor]);
  }
  if (!missing.length) return "";

  const earliest = Math.min(...missing);
  const sourceGlyph = ordinalGlyph(earliest);
  if (earliest <= 1) {
    return `${sourceGlyph} has no saved output. Use Run to start from the top.`;
  }
  return `${sourceGlyph} has no saved output. Start from ${ordinalGlyph(earliest - 1)} or earlier.`;
}

/**
 * Every enabled step that reads from a **disabled** one with no saved output — the editor's copy
 * of `workflow_graph.validate`'s rule (spec, "Execution order": a disabled `from` is read through
 * its saved handle, otherwise the graph fails validation). Surfacing it here is what stops the
 * user switching a step off and pressing Run into a pre-flight refusal.
 */
export function disabledSourceProblems(
  graph: WorkflowGraph,
  state: WorkflowState | null | undefined
): string[] {
  const byId = new Map(graph.nodes.map((candidate) => [candidate.id, candidate]));
  const positions = ordinals(graph);
  const problems: string[] = [];
  for (const node of graph.nodes) {
    if (!node.enabled || !node.from) continue;
    const source = byId.get(node.from);
    if (!source || source.enabled || hasSavedOutput(state, source.id)) continue;
    const self = ordinalGlyph(positions[node.id]);
    const from = ordinalGlyph(positions[source.id]);
    problems.push(
      `${self} reads from ${from}, which is disabled and has no saved output. Enable ${from} or point ${self} at another step.`
    );
  }
  return problems;
}

/**
 * Why `Move up` / `Move down` is disabled, or `""` when it is allowed.
 *
 * Both messages name the repair rather than the rule, because "reads from an earlier node" is the
 * invariant and "move the other one first" is the thing the user can actually do.
 */
export function moveBlockReason(
  graph: WorkflowGraph,
  nodeId: string,
  direction: MoveDirection
): string {
  if (canMove(graph, nodeId, direction)) return "";

  const index = graph.nodes.findIndex((node) => node.id === nodeId);
  if (index < 0) return "";
  const positions = ordinals(graph);
  const self = ordinalGlyph(positions[nodeId]);

  if (direction === "up") {
    if (index === 0) return "Already the first step.";
    const blocker = graph.nodes[index - 1];
    return `${self} reads from ${ordinalGlyph(positions[blocker.id])}. Move ${ordinalGlyph(
      positions[blocker.id]
    )} up first.`;
  }
  if (index >= graph.nodes.length - 1) return "Already the last step.";
  const blocker = graph.nodes[index + 1];
  const blockerGlyph = ordinalGlyph(positions[blocker.id]);
  return `${blockerGlyph} reads from ${self}. Move ${blockerGlyph} down first.`;
}

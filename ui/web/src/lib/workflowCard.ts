/**
 * The one-line prose a workflow card shows: what a step is configured to do, what the chain does
 * as a whole, and how long ago something happened.
 *
 * The *output* sentence is not here — `workflowNodeTypes.describeOutput` already owns it, and the
 * card, the add menu and the drawer's Output tab all read it from there. This module is only the
 * **input side**: the models, thresholds and folders the user typed, condensed.
 */

import { nodeTypeLabel } from "./workflowNodeTypes";
import type { WorkflowGraph, WorkflowNode, WorkflowState } from "../types/workflow";

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function list(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => text(item)).filter(Boolean) : [];
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** `D:/datasets/aoi · sidecar .txt` — the summary line under a card's title. */
export function nodeConfigSummary(node: WorkflowNode): string {
  const config = node.config ?? {};
  const parts: string[] = [];

  switch (node.type) {
    case "folder": {
      parts.push(text(config.path) || "no folder yet");
      const format = text(config.caption_format) || "sidecar";
      parts.push(format === "json" ? "captions in JSON" : `sidecar ${text(config.caption_ext) || ".txt"}`);
      break;
    }
    case "prep.tag": {
      const models = list(config.models);
      parts.push(models.length ? models.join(" + ") : "no tagger selected");
      const maxTags = num(config.max_tags);
      if (maxTags != null) parts.push(`max ${maxTags}`);
      if (config.overwrite) parts.push("overwrite");
      break;
    }
    case "prep.caption": {
      parts.push(text(config.model) || "no model selected");
      const prompt = text(config.prompt);
      parts.push(prompt ? "custom prompt" : text(config.prompt_base) || "descriptive-long");
      if (config.overwrite) parts.push("overwrite");
      break;
    }
    case "prep.clean": {
      const confidence = num(config.confidence);
      if (confidence != null) parts.push(`confidence ${confidence}`);
      parts.push(config.in_place ? "in place" : "into a new folder");
      // The form's own warning is longer; the card carries the short form of the same hazard.
      if (config.copy_undetected === false) parts.push("undetected images left behind");
      break;
    }
    case "prep.quality": {
      const metric = text(config.metric) || "blur";
      parts.push(metric);
      if (metric === "blur") {
        const threshold = num(config.blur_threshold);
        if (threshold != null) parts.push(`blur < ${threshold}`);
      } else if (metric === "iqa") {
        parts.push(text(config.iqa_model) || "clipiqa");
      } else {
        parts.push(`below ${text(config.aesthetic_min_label) || "normal"}`);
      }
      parts.push(text(config.action) === "move" ? "quarantine flagged" : "report only");
      break;
    }
    case "prep.index": {
      const models = list(config.models);
      parts.push(models.length ? models.join(" + ") : "no scorer selected");
      break;
    }
    case "tool": {
      parts.push(text(config.tool_id) || "no tool selected");
      const values = config.values;
      const count =
        values && typeof values === "object" ? Object.keys(values as object).length : 0;
      if (count) parts.push(`${count} ${count === 1 ? "input" : "inputs"}`);
      break;
    }
    case "train": {
      const jobId = config.job_id;
      parts.push(jobId == null || jobId === "" ? "no run selected" : `run #${jobId}`);
      break;
    }
    default:
      parts.push(`unknown step type ${node.type}`);
      break;
  }

  if (!node.gpu?.required) parts.push("CPU");
  else if (node.gpu.device != null) parts.push(`GPU ${node.gpu.device}`);
  else parts.push("GPU");

  return parts.filter(Boolean).join(" · ");
}

/**
 * `Source folder -> Tag -> Quality filter` — the list view's chain summary.
 *
 * Built from `WorkflowSummary.chain` (node types, in order) rather than titles: the list payload
 * carries only types (`_workflow_summary` avoids the N+1 of fetching every row's full graph just to
 * read its titles), so a step reads as its type label instead of whatever the user renamed it to.
 */
export function chainSummary(chain: string[], limit = 4): string {
  const labels = chain.map((type) => nodeTypeLabel(type));
  if (!labels.length) return "No steps yet";
  if (labels.length <= limit) return labels.join(" → ");
  return `${labels.slice(0, limit).join(" → ")} → +${labels.length - limit} more`;
}

/**
 * The handle the workflow currently *results in*: the last enabled node with a saved output.
 *
 * Not the last node — a `train` node emits nothing, and a disabled tail is not the result either.
 */
export function workflowResultPath(
  graph: Pick<WorkflowGraph, "nodes">,
  state: WorkflowState | null | undefined
): string {
  for (let index = graph.nodes.length - 1; index >= 0; index -= 1) {
    const node = graph.nodes[index];
    if (!node.enabled) continue;
    const output = state?.nodes?.[node.id]?.output;
    if (output?.path) return output.path;
  }
  return "";
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * `2h ago` / `just now` / `3d ago`.
 *
 * The server sends `datetime.now(timezone.utc).isoformat()` — offset-aware, so it parses
 * unambiguously. A **naive** timestamp (no offset, from older rows) would be read as local time and
 * could land in the future; that case is clamped to "just now" rather than shown as "-3h ago".
 */
export function relativeTime(iso: unknown, now: number = Date.now()): string {
  if (typeof iso !== "string" || !iso) return "";
  const at = new Date(iso);
  const stamp = at.getTime();
  if (Number.isNaN(stamp)) return "";

  const delta = now - stamp;
  if (delta < MINUTE) return "just now";
  if (delta < HOUR) return `${Math.floor(delta / MINUTE)}m ago`;
  if (delta < DAY) return `${Math.floor(delta / HOUR)}h ago`;
  return `${Math.floor(delta / DAY)}d ago`;
}

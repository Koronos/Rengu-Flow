/** Types for the Workflows feature.
 *
 * Derived from what the server actually sends, not the spec:
 * - graph model: rengu_flow_ui/workflow_graph.py (DatasetHandle, NodeGpu, Variable, WorkflowNode,
 *   WorkflowGraph, node_to_dict/graph_to_dict)
 * - run state: rengu_flow_ui/workflow_runner.py (the state machine + state_json shape)
 * - HTTP payloads: rengu_flow_ui/workflow_routes.py (_workflow_summary, _workflow_detail, request
 *   bodies)
 */

import type { RunProgress } from "./api";

// ------------------------------------------------------------------------------ graph model

/** The only value that travels between nodes (workflow_graph.DatasetHandle). Connecting is always valid. */
export interface DatasetHandle {
  path: string;
  caption_format: string;
  caption_ext: string;
}

/** Per-node GPU policy (workflow_graph.NodeGpu). `device` is a physical index; `null` is auto. */
export interface WorkflowNodeGpu {
  required: boolean;
  wait: boolean;
  device: number | null;
}

/** A workflow-level string constant (workflow_graph.Variable). Configuration only, never node output. */
export interface WorkflowVariable {
  name: string;
  value: string;
  description: string;
}

/**
 * One executable step (workflow_graph.WorkflowNode, serialized by node_to_dict).
 *
 * The wire key is literally `from` — Python maps its `source` field to/from it because `from` is
 * a keyword there, but `from` is a perfectly valid TypeScript property name, so it is kept as-is.
 */
export interface WorkflowNode {
  id: string;
  type: string;
  title: string;
  from: string | null;
  enabled: boolean;
  config: Record<string, unknown>;
  gpu: WorkflowNodeGpu;
}

/** The saved graph (workflow_graph.WorkflowGraph, serialized by graph_to_dict). */
export interface WorkflowGraph {
  version: number;
  name: string;
  description: string;
  variables: WorkflowVariable[];
  nodes: WorkflowNode[];
}

// ------------------------------------------------------------------------------ run state

/**
 * `pending -> waiting_gpu -> launching -> running -> done | failed`, with `running -> stopping ->
 * stopped` for a cancel, and `pending -> skipped` for a disabled node. See the state-machine
 * diagram in workflow_runner.py's module docstring.
 */
export type NodeStatus =
  | "pending"
  | "waiting_gpu"
  | "launching"
  | "running"
  | "stopping"
  | "done"
  | "failed"
  | "stopped"
  | "skipped";

/**
 * `idle` is what a fresh workflow's `state_json` (`'{}'`) reads as — workflow_routes._workflow_summary
 * defaults to it explicitly. The rest are written by workflow_runner (`running`, `cancelling`,
 * `done`, `failed`, `stopped`).
 */
export type WorkflowStatus = "idle" | "running" | "cancelling" | "done" | "failed" | "stopped";

/**
 * One node's entry in `WorkflowState.nodes[id]` (workflow_runner._update_node and friends).
 *
 * A node that has never run has no entry at all, and the fields below are written incrementally
 * by different call sites (`_mark_launching`, `_complete_node`, `_fail_node`, `_stop_node`,
 * reconciliation's adoption path) — nothing here is guaranteed present at once.
 */
export interface NodeState {
  status?: NodeStatus;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  error?: string;
  pid?: number | null;
  pid_create_time?: number | null;
  /** Set when a restart adopted a `launching` node with no pid because its log was still growing. */
  adopted?: boolean;
  stop_requested_at?: number | null;
  /** The handle this node produced. Cleared on failure/stop — a downstream node never reads a stale success. */
  output?: DatasetHandle | null;
  /** The upstream handle this node actually consumed; workflow_graph.compute_stale compares against it. */
  saved_input?: DatasetHandle | null;
  /** Digest over (type, materialized config, gpu, parent hash) — how staleness is detected. */
  config_hash?: string;
  /** Present for inline nodes (`train`) — e.g. `{ job_id }`. */
  result?: unknown;
}

/** The workflow's single saved run state (the `state_json` column, parsed). */
export interface WorkflowState {
  status?: WorkflowStatus;
  current_node?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  nodes?: Record<string, NodeState>;
  /** The standing front-of-queue claim a `train` node holds (workflow_runner._settle_queue_claim). */
  queue_claim?: { job_id: number | string } | null;
}

// ------------------------------------------------------------------------------ HTTP payloads

/** One row of `GET /workflows` (workflow_routes._workflow_summary) — no graph, for the list view. */
export interface WorkflowSummary {
  id: number | string;
  name: string;
  /** Every node's type, in list order (includes disabled nodes). `chain.length === steps`. */
  chain: string[];
  steps: number;
  status: WorkflowStatus;
  updated_at: string;
}

export interface WorkflowListResult {
  workflows: WorkflowSummary[];
}

/** The editor's full load payload (workflow_routes._workflow_detail): graph + state + version. */
export interface WorkflowDetail {
  id: number | string;
  name: string;
  graph: WorkflowGraph;
  state: WorkflowState;
  /** Node id -> whether its saved output/hash no longer matches its current configuration. */
  stale: Record<string, boolean>;
  /** Optimistic-concurrency token: send back as `version` on `PUT`, get a fresh one in the response. */
  version: number;
  created_at: string;
  updated_at: string;
}

/** Body for `PUT /workflows/{id}` (workflow_routes.UpdateGraphBody). */
export interface WorkflowUpdatePayload {
  graph: WorkflowGraph;
  version: number;
}

export interface WorkflowValidateResult {
  errors: string[];
}

/** Body for `POST /workflows/{id}/start` (workflow_routes.StartWorkflowBody). */
export interface WorkflowStartOptions {
  /** Re-run this node and everything downstream, reusing earlier nodes' saved output. */
  from_node?: string | null;
  /** Re-run even nodes that are `done` and not stale. */
  force?: boolean;
  /**
   * With `from_node`: run *that node alone*, leaving its descendants exactly as they are.
   * `only: true` without `from_node` is a 400 — never send one without the other.
   */
  only?: boolean;
}

export interface WorkflowNodeLogResult {
  chunk: string;
  offset: number;
  /**
   * The last complete progress-marker line in the node's log tail, if any. Same wire shape as a
   * training run's progress: both come from control/progress_stream.parse_last_progress_marker.
   */
  progress: RunProgress | null;
}

/**
 * `GET /workflows/{id}/nodes/{node_id}/report` (workflow_routes.node_report_route) — the node's
 * structured result, read from disk. `report` is whatever JSON the step wrote: a prep stage always
 * writes an object, but a tool can return any JSON value (a list, a string, a number), which is why
 * it travels wrapped rather than as the whole response body.
 */
export interface WorkflowNodeReportResult {
  file: "report.json" | "result.json";
  report: unknown;
}

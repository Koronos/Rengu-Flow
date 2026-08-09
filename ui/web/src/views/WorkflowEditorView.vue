<template>
  <div class="workflow-editor page-shell">
    <WorkflowRunBar
      :name="graph.name || editor.name.value"
      :status="workflowStatus"
      :running="running"
      :busy="run.busy.value"
      :read-only="readOnly"
      :run-count="runSet.length"
      :blocked-reason="blockedReason"
      :variable-count="graph.variables.length"
      :undefined-count="undefinedNames.length"
      :result-path="resultPath"
      :last-run-label="lastRunLabel"
      :saving="editor.saving.value"
      :dirty="editor.dirty.value"
      :stream-status="run.streamStatus.value"
      @back="router.push('/workflows')"
      @run="runAll(false)"
      @run-force="runAll(true)"
      @validate="validateOnly"
      @stop="stopRun"
      @variables="variablesOpen = true"
      @duplicate="duplicateWorkflow"
      @delete="deleteWorkflow"
      @rename="editor.setName"
    />

    <!--
      A 409 is the one error in this app that must not be a toast. Workflows have no history, so
      the local graph is the only copy of the user's last edits and reloading discards them: the
      choice belongs to the user, in a banner that stays put.
    -->
    <el-alert
      v-if="editor.conflict.value"
      type="warning"
      show-icon
      :closable="false"
      class="mb-12"
      title="This workflow changed elsewhere"
    >
      <p class="wf-alert__body">{{ editor.conflict.value }}</p>
      <p class="wf-alert__body">
        Your unsaved edits are still on screen. Reloading replaces them with the server's copy —
        workflows keep no history, so copy anything you need first.
      </p>
      <el-button size="small" type="warning" @click="editor.reload">Reload</el-button>
    </el-alert>

    <el-alert
      v-if="readOnly"
      type="info"
      show-icon
      :closable="false"
      class="mb-12"
      title="Stop to edit"
      description="The runner owns this workflow while it is running; editing resumes once it stops."
    />

    <el-alert
      v-if="editor.error.value"
      type="error"
      show-icon
      :closable="false"
      class="mb-12"
      :title="editor.error.value"
    />

    <el-alert
      v-if="run.preflight.value.length"
      type="error"
      show-icon
      class="mb-12"
      :title="`Pre-flight found ${run.preflight.value.length} problem${run.preflight.value.length === 1 ? '' : 's'}`"
      @close="run.preflight.value = []"
    >
      <ul class="wf-preflight">
        <li v-for="(problem, index) in run.preflight.value" :key="index">{{ problem }}</li>
      </ul>
    </el-alert>

    <div v-loading="editor.loading.value" class="wf-chain" @mouseleave="clearHover">
      <el-empty
        v-if="!editor.loading.value && !graph.nodes.length"
        description="No steps yet — a workflow starts with a source folder"
        :image-size="64"
      />

      <div
        v-for="(node, index) in graph.nodes"
        :id="rowDomId(node.id)"
        :key="node.id"
        class="wf-row"
        :class="{ 'wf-row--last': index === graph.nodes.length - 1 }"
      >
        <WorkflowEdgeGutter
          :segments="gutter[index] ?? []"
          :active-keys="activeKeys"
          :dimmed="dimming"
        />

        <WorkflowNodeCard
          class="wf-row__card"
          :node="node"
          :position="index + 1"
          :chip="chipFor(node)"
          :summary="nodeConfigSummary(node)"
          :output-sentence="describeOutput(node)"
          :stale="Boolean(staleMap[node.id])"
          :percent="percentFor(node.id)"
          :jump="jumpFor(node)"
          :highlight="highlightFor(node.id)"
          :dimmed="dimmedFor(node.id)"
          :skipped-note="skippedNoteFor(node.id)"
          :active="selectedNodeId === node.id"
          :read-only="readOnly"
          :run-from-reason="runFromBlockReason(graph, node.id, state)"
          :up-reason="moveBlockReason(graph, node.id, 'up')"
          :down-reason="moveBlockReason(graph, node.id, 'down')"
          @open="openNode(node.id)"
          @card-enter="hoverNodeId = node.id"
          @card-leave="hoverNodeId = null"
          @badge-enter="focusJump(node)"
          @badge-leave="focusEdge = null"
          @badge-activate="scrollToSource(node)"
          @run-from="runFrom(node.id)"
          @run-only="runNode(node.id)"
          @move="(direction) => moveStep(node.id, direction)"
          @toggle-enabled="toggleStep(node.id)"
          @delete="deleteStep(node.id)"
        />

        <!-- The hairline "insert here" affordance, in the gap this row already reserves. -->
        <div v-if="index < graph.nodes.length - 1" class="wf-row__insert">
          <WorkflowAddStepPopover
            :tools="tools"
            :disabled="readOnly"
            @select="(choice) => addStep(index + 1, choice, true)"
          >
            <el-button
              size="small"
              circle
              :icon="Plus"
              :disabled="readOnly"
              v-bind="ariaLabel(`Insert a step after step ${index + 1}`)"
            />
          </WorkflowAddStepPopover>
        </div>
      </div>

      <div class="wf-chain__foot">
        <WorkflowAddStepPopover
          :tools="tools"
          :disabled="readOnly"
          placement="top"
          @select="(choice) => addStep(graph.nodes.length, choice, false)"
        >
          <el-button :icon="Plus" :disabled="readOnly">Add step</el-button>
        </WorkflowAddStepPopover>
      </div>
    </div>

    <WorkflowVariablesDialog
      v-model:open="variablesOpen"
      :graph="graph"
      @update:variables="onVariablesChange"
    />

    <WorkflowNodeDrawer
      v-model:open="drawerOpen"
      :node="selectedNode"
      :graph="graph"
      :state="state"
      :stale="staleMap"
      :workflow-id="workflowId"
      :read-only="readOnly"
      @update:node="onNodeUpdate"
      @run-node="runNode"
      @run-from="runFrom"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import { Plus } from "@element-plus/icons-vue";
import { api, type ToolboxToolSummary } from "../api";
import { ariaLabel } from "../lib/aria";
import { formatError } from "../lib/formatError";
import {
  addNode,
  createNode,
  moveNode,
  ordinalGlyph,
  ordinals,
  removeNode,
  type MoveDirection,
} from "../lib/workflowGraph";
import { gutterRows } from "../lib/workflowGutter";
import { edgeKey, isJump, skippedBy } from "../lib/workflowLayout";
import { describeOutput } from "../lib/workflowNodeTypes";
import { nodeConfigSummary, relativeTime, workflowResultPath } from "../lib/workflowCard";
import {
  moveBlockReason,
  nodeChip,
  nodeEntry,
  nodesToRun,
  runFromBlockReason,
} from "../lib/workflowStatus";
import { unknownRefs } from "../lib/workflowVars";
import { useWorkflowEditor } from "../composables/useWorkflowEditor";
import { useWorkflowRun } from "../composables/useWorkflowRun";
import WorkflowAddStepPopover from "../components/workflow/WorkflowAddStepPopover.vue";
import WorkflowEdgeGutter from "../components/workflow/WorkflowEdgeGutter.vue";
import WorkflowNodeCard from "../components/workflow/WorkflowNodeCard.vue";
import WorkflowNodeDrawer from "../components/workflow/WorkflowNodeDrawer.vue";
import WorkflowRunBar from "../components/workflow/WorkflowRunBar.vue";
import WorkflowVariablesDialog from "../components/workflow/WorkflowVariablesDialog.vue";
import type { WorkflowGraph, WorkflowNode, WorkflowVariable } from "../types/workflow";

const vLoading = ElLoadingDirective;

const route = useRoute();
const router = useRouter();

const workflowId = computed(() => String(route.params.id ?? ""));

const editor = useWorkflowEditor(workflowId);
const graph = computed(() => editor.graph.value);
const state = computed(() => editor.state.value);
const staleMap = computed(() => editor.stale.value);
const workflowStatus = computed(() => state.value.status ?? "idle");
const running = computed(() => workflowStatus.value === "running" || workflowStatus.value === "cancelling");
const readOnly = computed(() => editor.readOnly.value);

const run = useWorkflowRun({
  workflowId,
  running,
  currentNode: computed(() => state.value.current_node ?? null),
  onDetail: editor.applyLive,
});

// ------------------------------------------------------------------ chain geometry

const gutter = computed(() => gutterRows(graph.value.nodes));
const positions = computed(() => ordinals(graph.value));

/**
 * Hover state lives here, not in the cards, because one hover repaints several of them: the badge
 * highlight has to dim the cards it is *not* about, which no single card can know.
 */
const hoverNodeId = ref<string | null>(null);
const focusEdge = ref<{ key: string; source: string; target: string; skipped: string[] } | null>(null);

function clearHover(): void {
  hoverNodeId.value = null;
  focusEdge.value = null;
}

const activeKeys = computed<string[]>(() => {
  if (focusEdge.value) return [focusEdge.value.key];
  const id = hoverNodeId.value;
  if (!id) return [];
  const keys: string[] = [];
  for (const node of graph.value.nodes) {
    if (node.from === id) keys.push(edgeKey(id, node.id));
    if (node.id === id && node.from) keys.push(edgeKey(node.from, id));
  }
  return keys;
});

const dimming = computed(() => focusEdge.value !== null);

/** `null` for a consecutive link: the connector already says it, so no badge is drawn. */
function jumpFor(node: WorkflowNode): { key: string; sourcePosition: number } | null {
  if (!node.from || !isJump(graph.value.nodes, node.id)) return null;
  return { key: edgeKey(node.from, node.id), sourcePosition: positions.value[node.from] };
}

function focusJump(node: WorkflowNode): void {
  if (!node.from) return;
  focusEdge.value = {
    key: edgeKey(node.from, node.id),
    source: node.from,
    target: node.id,
    skipped: skippedBy(graph.value.nodes, node.id),
  };
}

function highlightFor(nodeId: string): "source" | "target" | "none" {
  const edge = focusEdge.value;
  if (!edge) return "none";
  if (edge.source === nodeId) return "source";
  if (edge.target === nodeId) return "target";
  return "none";
}

function dimmedFor(nodeId: string): boolean {
  const edge = focusEdge.value;
  return edge !== null && edge.source !== nodeId && edge.target !== nodeId;
}

function skippedNoteFor(nodeId: string): string {
  const edge = focusEdge.value;
  if (!edge || !edge.skipped.includes(nodeId)) return "";
  return `${ordinalGlyph(positions.value[edge.target])} reads past this step`;
}

function rowDomId(nodeId: string): string {
  return `wf-row-${nodeId}`;
}

function scrollToSource(node: WorkflowNode): void {
  if (!node.from) return;
  const element = document.getElementById(rowDomId(node.from));
  element?.scrollIntoView({ behavior: "smooth", block: "center" });
  focusJump(node);
}

// ------------------------------------------------------------------ per-node presentation

function chipFor(node: WorkflowNode) {
  return nodeChip(node, nodeEntry(state.value, node.id), workflowStatus.value);
}

/** Progress belongs to the node the runner is on; no other card may claim it. */
function percentFor(nodeId: string): number | null {
  if (state.value.current_node !== nodeId) return null;
  return run.progress.value?.percent ?? null;
}

// ------------------------------------------------------------------ run bar inputs

const runSet = computed(() => nodesToRun(graph.value, state.value, staleMap.value));
const undefinedNames = computed(() => unknownRefs(graph.value));
const resultPath = computed(() => workflowResultPath(graph.value, state.value));

const blockedReason = computed(() => {
  if (!graph.value.nodes.length) return "Add a step first.";
  if (undefinedNames.value.length) {
    return `Undefined ${undefinedNames.value.length === 1 ? "variable" : "variables"}: ${undefinedNames.value
      .map((name) => `\${${name}}`)
      .join(", ")}. Open Variables to give them a value.`;
  }
  return "";
});

const lastRunLabel = computed(() => {
  if (running.value && state.value.started_at) {
    return `started ${relativeTime(state.value.started_at)}`;
  }
  const finished = relativeTime(state.value.finished_at);
  return finished ? `last run ${finished}` : "";
});

// ------------------------------------------------------------------ editing

const tools = ref<ToolboxToolSummary[]>([]);

function addStep(
  at: number,
  choice: { type: string; title: string; config?: Record<string, unknown> },
  splice: boolean
): void {
  editor.mutate((current) =>
    addNode(current, createNode(choice.type, { title: choice.title, config: choice.config }), {
      at,
      splice,
    })
  );
}

function moveStep(id: string, direction: MoveDirection): void {
  editor.mutate((current) => moveNode(current, id, direction));
}

function toggleStep(id: string): void {
  editor.mutate((current) => ({
    ...current,
    nodes: current.nodes.map((node) =>
      node.id === id ? { ...node, enabled: !node.enabled } : node
    ),
  }));
}

/**
 * Delete, with the prompt the situation deserves.
 *
 * An unreferenced node goes straight away. A referenced one is spliced by default — children
 * inherit the deleted node's `from` — which is the least surprising repair. A `folder` is the
 * exception: its `from` is `null`, so splicing leaves its children sourceless, and saying so up
 * front beats a pre-flight error the user then has to decode.
 */
async function deleteStep(id: string): Promise<void> {
  const current = graph.value;
  const node = current.nodes.find((candidate) => candidate.id === id);
  if (!node) return;
  const children = current.nodes.filter((candidate) => candidate.from === id);
  const self = ordinalGlyph(positions.value[id]);

  if (children.length) {
    const childList = children.map((child) => ordinalGlyph(positions.value[child.id])).join(", ");
    const message =
      node.type === "folder"
        ? `${childList} read from ${self}. Deleting it leaves ${
            children.length === 1 ? "that step" : "those steps"
          } with no source — pick a new source folder before running.`
        : `${childList} read from ${self}. Deleting it connects ${
            children.length === 1 ? "it" : "them"
          } to ${
            node.from ? ordinalGlyph(positions.value[node.from]) : "nothing"
          } instead. Files already written are left on disk.`;
    try {
      await ElMessageBox.confirm(message, `Delete step ${self}?`, {
        type: "warning",
        confirmButtonText: "Delete and reconnect",
      });
    } catch {
      return;
    }
  }

  if (selectedNodeId.value === id) closeDrawer();
  editor.mutate((currentGraph) => removeNode(currentGraph, id));
}

function onNodeUpdate(node: WorkflowNode): void {
  editor.mutate((current) => ({
    ...current,
    nodes: current.nodes.map((candidate) => (candidate.id === node.id ? node : candidate)),
  }));
}

function onVariablesChange(variables: WorkflowVariable[]): void {
  editor.mutate((current: WorkflowGraph) => ({ ...current, variables }));
}

// ------------------------------------------------------------------ drawer + deep link

const variablesOpen = ref(false);
const selectedNodeId = ref<string | null>(null);
const drawerOpen = computed({
  get: () => selectedNodeId.value !== null,
  set: (value: boolean) => {
    if (!value) closeDrawer();
  },
});
const selectedNode = computed(
  () => graph.value.nodes.find((node) => node.id === selectedNodeId.value) ?? null
);

function openNode(id: string): void {
  selectedNodeId.value = id;
  void router.replace({ query: { ...route.query, node: id } });
}

function closeDrawer(): void {
  selectedNodeId.value = null;
  const query = { ...route.query };
  delete query.node;
  void router.replace({ query });
}

// `?node=n2` makes a step linkable; the graph may load after the route, so this waits for both.
watch(
  [() => route.query.node, () => graph.value.nodes],
  ([queryNode, nodes]) => {
    const wanted = typeof queryNode === "string" ? queryNode : null;
    if (!wanted) {
      selectedNodeId.value = null;
      return;
    }
    selectedNodeId.value = nodes.some((node) => node.id === wanted) ? wanted : null;
  },
  { immediate: true }
);

// ------------------------------------------------------------------ running

/**
 * Every run flushes first: the server runs what is *saved*, not what is on screen.
 *
 * A refused flush therefore refuses the run — and says so. `save()` also returns `false` for a
 * write that is merely still in flight, so silence here reads as a dead button: the user presses
 * Run during the 700 ms autosave and nothing whatsoever happens.
 */
async function withSavedGraph(action: () => Promise<unknown>): Promise<void> {
  const saved = await editor.flush();
  if (!saved) {
    ElMessage.warning(
      editor.saving.value
        ? "Still saving your last edit — press Run again in a moment."
        : "Your last edit could not be saved, so nothing was started. See the message above.",
    );
    return;
  }
  await action();
}

async function runAll(force: boolean): Promise<void> {
  await withSavedGraph(() => run.start({ force }));
}

async function runFrom(nodeId: string): Promise<void> {
  await withSavedGraph(() => run.start({ from_node: nodeId }));
}

/**
 * "Run only this step": the server runs `from_node` alone, leaving its descendants exactly as they
 * are — no reset, no re-run. `_require_saved_ancestors` still applies, so a step with no saved
 * upstream input is refused the same way `runFrom` is.
 */
async function runNode(nodeId: string): Promise<void> {
  await withSavedGraph(() => run.start({ from_node: nodeId, only: true }));
}

async function stopRun(): Promise<void> {
  await run.stop();
}

async function validateOnly(): Promise<void> {
  await withSavedGraph(() => run.validate());
}

// ------------------------------------------------------------------ workflow-level actions

async function duplicateWorkflow(): Promise<void> {
  try {
    // The server clones what it has. A refused flush (a run holds the workflow, or the save
    // 409'd) means the newest edit is not part of it, and copying it away in silence would hand
    // the user a duplicate they believe is current.
    if (!(await editor.flush())) {
      ElMessage.warning("Duplicating the last saved version — your newest edit is not in it yet.");
    }
    const clone = await api.cloneWorkflow(workflowId.value);
    ElMessage.success(`Duplicated as #${clone.id}`);
    void router.push(`/workflows/${clone.id}`);
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function deleteWorkflow(): Promise<void> {
  try {
    await ElMessageBox.confirm(
      "Deleting a workflow removes its graph and its run history. Files already written to disk are left alone.",
      `Delete "${graph.value.name || editor.name.value || workflowId.value}"?`,
      { type: "warning" }
    );
  } catch {
    return;
  }
  try {
    await api.deleteWorkflow(workflowId.value);
    ElMessage.success("Deleted");
    void router.push("/workflows");
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

onMounted(async () => {
  try {
    tools.value = await api.listToolboxTools();
  } catch {
    // The add menu degrades to "No saved tools yet"; nothing else depends on the list.
  }
});
</script>

<style scoped>
.workflow-editor {
  max-width: 980px;
}

.wf-chain {
  min-height: 160px;
}

.wf-row {
  position: relative;
  display: flex;
  align-items: stretch;
  gap: 0;
}

/*
 * The gap between cards is the card's own margin, NOT the row's padding: a flex line's cross size
 * counts its items' margins but not the container's padding, so only this way does the stretched
 * gutter reach across the gap and the connector stay unbroken between two cards.
 */
.wf-row__card {
  flex: 1 1 auto;
  min-width: 0;
  margin-bottom: 24px;
}

.wf-row--last .wf-row__card {
  margin-bottom: 0;
}

.wf-row__insert {
  position: absolute;
  left: 28px;
  right: 0;
  bottom: 0;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
}

.wf-row:hover .wf-row__insert,
.wf-row:focus-within .wf-row__insert {
  opacity: 1;
  pointer-events: auto;
}

.wf-chain__foot {
  display: flex;
  justify-content: center;
  padding: 20px 0 8px;
}

.wf-alert__body {
  margin: 0 0 8px;
  line-height: 1.5;
}

.wf-preflight {
  margin: 0;
  padding-left: 18px;
  line-height: 1.6;
}
</style>

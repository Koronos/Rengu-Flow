<!--
  One node, opened from its card in the editor.

  The header is **persistent across tabs** and carries the progress bar: progress belongs where it
  is visible while the user edits configuration, not behind a fifth tab they would have to leave
  the form to look at.

  | Tab | What it answers |
  |---|---|
  | Configure | what this step does, and how it runs |
  | Input | *which folder actually goes in* — the antidote to getting lost among jumps |
  | Output | what comes out, and where the result of the last run went |
  | Logs | what it is printing right now |

  The Input tab is the load-bearing one. A chain with a jump in it ("④ reads ②, not ③") is exactly
  where a user loses track, so the tab states the folder outright and says whether that folder is
  **saved** (the source really produced it) or **predicted** (the source has not run, so this is
  what it *would* emit) — never blurring the two, because acting on a prediction as if it were a
  fact is how a stage gets pointed at a folder that does not exist yet.
-->
<template>
  <el-drawer
    :model-value="open"
    direction="rtl"
    :size="isMobile ? '100%' : '640px'"
    :with-header="false"
    class="node-drawer"
    @update:model-value="emit('update:open', $event)"
  >
    <div v-if="node" class="node-drawer__body">
      <!-- Persistent header ---------------------------------------------------- -->
      <header class="node-drawer__head">
        <div class="node-drawer__title-row">
          <span class="node-drawer__ordinal">{{ ordinalGlyph(ordinal) }}</span>
          <div class="node-drawer__title">
            <span class="node-drawer__name">{{ node.title }}</span>
            <span class="node-drawer__type">{{ nodeTypeLabel(node.type) }}</span>
          </div>

          <el-tag :type="statusChip.type" size="small" :effect="statusChip.effect">
            {{ statusChip.label }}
          </el-tag>
          <el-tag v-if="isStale" type="warning" size="small" effect="plain">Stale</el-tag>

          <el-button
            type="primary"
            size="small"
            :icon="CaretRight"
            :disabled="!node.enabled || readOnly"
            @click="emit('run-node', node.id)"
          >
            Run
          </el-button>

          <el-dropdown trigger="click" @command="onCommand">
            <el-button size="small" text :icon="MoreFilled" v-bind="ariaLabel('Node actions')" />
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item command="run-node" :disabled="!node.enabled || readOnly">
                  Run only this step
                </el-dropdown-item>
                <el-dropdown-item command="run-from" :disabled="!node.enabled || readOnly">
                  Run from here
                </el-dropdown-item>
                <el-dropdown-item command="rename" divided :disabled="readOnly">Rename…</el-dropdown-item>
                <el-dropdown-item command="toggle-enabled" :disabled="readOnly">
                  {{ node.enabled ? "Disable step" : "Enable step" }}
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>

          <el-button size="small" text :icon="Close" v-bind="ariaLabel('Close')" @click="close" />
        </div>

        <!-- Progress lives in the header so it stays visible on every tab. -->
        <div class="node-drawer__progress">
          <el-progress
            v-if="percent != null"
            :percentage="percent"
            :status="progressStatus"
            :stroke-width="6"
            striped
            :striped-flow="isLive"
          />
          <div class="node-drawer__progress-meta">
            <el-text size="small" type="info">{{ progressLabel }}</el-text>
            <el-button
              v-if="isLive"
              size="small"
              type="danger"
              plain
              :loading="stopping"
              @click="stop"
            >
              Stop
            </el-button>
          </div>
        </div>
      </header>

      <!--
        The same banner the editor page shows, in the one place the page's own is out of sight.
        Every control below is disabled with it: the runner owns the workflow while it runs, so a
        keystroke here would be dropped by `useWorkflowEditor.mutate` and silently revert on the
        next open. Being told "Stop to edit" costs the user nothing; losing an edit costs them the
        only copy.
      -->
      <el-alert
        v-if="readOnly"
        type="info"
        show-icon
        :closable="false"
        class="node-drawer__readonly"
        title="Stop to edit"
        description="The runner owns this workflow while it is running; editing resumes once it stops."
      />

      <el-tabs v-model="tab" class="node-drawer__tabs">
        <!-- Configure ---------------------------------------------------------- -->
        <el-tab-pane label="Configure" name="configure">
          <NodeRuntimeFields
            v-if="node.type !== 'folder'"
            :model-value="node"
            :graph="graph"
            :source-paths="sourcePaths"
            :disabled="readOnly"
            @update:model-value="emit('update:node', $event)"
          />
          <el-divider v-if="node.type !== 'folder'" />

          <FolderNodeForm
            v-if="node.type === 'folder'"
            :model-value="node.config"
            :disabled="readOnly"
            @update:model-value="patchConfig"
          />
          <ToolNodeForm
            v-else-if="node.type === 'tool'"
            :model-value="node.config"
            :disabled="readOnly"
            @update:model-value="patchConfig"
          />
          <TrainNodeForm
            v-else-if="node.type === 'train'"
            :model-value="node.config"
            :queued-job-id="queuedJobId"
            :disabled="readOnly"
            @update:model-value="patchConfig"
          />

          <template v-else-if="prepStage">
            <!--
              `hide-dataset-fields` is the whole point of the extraction: in a workflow the folder
              and caption layout come from the incoming edge, so the stage form contributes only
              the stage's own settings. It renders nothing today and is kept so a future
              non-dataset common field lands here for free.
            -->
            <PrepCommonFields
              v-model="commonForm"
              :stage="prepStage"
              hide-dataset-fields
              :disabled="readOnly"
            />

            <TagStageForm
              v-if="prepStage === 'tag'"
              :key="`tag-${node.id}`"
              v-model="tagForm"
              v-model:thresholds="tagThresholds"
              :seed="seedSection as PrepTagConfig | null"
              :disabled="readOnly"
              @models-loaded="tagModels = $event"
            />
            <CaptionStageForm
              v-else-if="prepStage === 'caption'"
              :key="`caption-${node.id}`"
              v-model="captionForm"
              v-model:prompt-options="promptOptions"
              v-model:preview-text="previewText"
              v-model:preview-native="previewNative"
              :seed="seedSection as PrepCaptionConfig | null"
              :disabled="readOnly"
            />
            <CleanStageForm
              v-else-if="prepStage === 'clean'"
              :key="`clean-${node.id}`"
              v-model="cleanForm"
              :seed="seedSection as PrepCleanConfig | null"
              :disabled="readOnly"
            />
            <QualityStageForm
              v-else-if="prepStage === 'quality'"
              :key="`quality-${node.id}`"
              v-model="qualityForm"
              :common-form="commonForm"
              :seed="seedSection as PrepQualityConfig | null"
              :disabled="readOnly"
            />
            <IndexStageForm
              v-else-if="prepStage === 'index'"
              :key="`index-${node.id}`"
              v-model="indexForm"
              :seed="seedSection as PrepIndexConfig | null"
              :disabled="readOnly"
            />
          </template>

          <el-alert
            v-else
            type="error"
            :closable="false"
            show-icon
            :title="`Unknown step type ${node.type}. It is kept exactly as saved, but this version cannot run or edit it.`"
          />
        </el-tab-pane>

        <!-- Input -------------------------------------------------------------- -->
        <el-tab-pane label="Input" name="input">
          <el-empty
            v-if="!consumes"
            description="This step is a source: it has no input, it defines one."
            :image-size="60"
          />
          <template v-else>
            <el-alert
              v-if="!node.from"
              type="error"
              :closable="false"
              show-icon
              title="No source. Pick one under Configure — the workflow refuses to start until every step has one."
            />
            <template v-else>
              <dl class="node-drawer__facts">
                <div class="node-drawer__fact">
                  <dt>Reads from</dt>
                  <dd>
                    {{ ordinalGlyph(sourceOrdinal) }} {{ sourceNode?.title ?? node.from }}
                    <el-tag v-if="isJump" size="small" type="info" effect="plain" class="ml-8">
                      jumps back {{ ordinal - sourceOrdinal }} steps
                    </el-tag>
                  </dd>
                </div>
                <div class="node-drawer__fact">
                  <dt>Folder</dt>
                  <dd>
                    <code class="node-drawer__path">{{ inputHandle?.path || "—" }}</code>
                    <el-tag
                      v-if="inputHandle"
                      size="small"
                      :type="inputIsSaved ? 'success' : 'warning'"
                      effect="plain"
                      class="ml-8"
                    >
                      {{ inputIsSaved ? "saved" : "predicted" }}
                    </el-tag>
                  </dd>
                </div>
                <div class="node-drawer__fact">
                  <dt>Captions</dt>
                  <dd>{{ captionLayoutLabel }}</dd>
                </div>
              </dl>

              <el-alert
                v-if="inputHandle && !inputIsSaved"
                type="warning"
                :closable="false"
                show-icon
                class="node-drawer__note"
                :title="`${sourceNode?.title ?? 'The source step'} has not run yet, so this folder is what it would emit, not what it did. A step that computes its output folder can still surprise you here.`"
              />
              <el-alert
                v-if="driftedInput"
                type="warning"
                :closable="false"
                show-icon
                class="node-drawer__note"
                :title="`Last run this step consumed ${driftedInput}. The folder above is different, so the saved result no longer matches its input.`"
              />

              <div class="node-drawer__stats">
                <PathValidationFeedback :loading="statsLoading" :error="statsError" />
                <el-text v-if="folderStats && folderStats.ok !== false" size="small" type="info">
                  {{ mediaSummary }}
                </el-text>
              </div>
            </template>
          </template>
        </el-tab-pane>

        <!-- Output ------------------------------------------------------------- -->
        <el-tab-pane label="Output" name="output">
          <dl class="node-drawer__facts">
            <div class="node-drawer__fact">
              <dt>Rule</dt>
              <dd>{{ describeOutput(node) }}</dd>
            </div>
            <div v-if="emits" class="node-drawer__fact">
              <dt>Emits</dt>
              <dd>
                <code class="node-drawer__path">{{ outputHandle?.path || "—" }}</code>
                <el-tag
                  v-if="outputHandle"
                  size="small"
                  :type="outputIsSaved ? 'success' : 'warning'"
                  effect="plain"
                  class="ml-8"
                >
                  {{ outputIsSaved ? "saved" : "predicted" }}
                </el-tag>
              </dd>
            </div>
          </dl>

          <el-card v-if="queuedJobId != null" shadow="never" class="node-drawer__queued">
            <router-link :to="`/runs/jobs/${queuedJobId}`" class="node-drawer__link">
              Queued run #{{ queuedJobId }} &rarr;
            </router-link>
            <el-text size="small" type="info" class="node-drawer__hint">
              This step is done because the run was queued — not because it trained.
            </el-text>
          </el-card>

          <el-alert
            v-if="nodeState?.error"
            type="error"
            :closable="false"
            show-icon
            class="node-drawer__note"
            :title="nodeState.error"
          />

          <template v-if="prepStage">
            <el-divider content-position="left">What this step is set to do</el-divider>
            <PrepJobSummaryPanel
              :stage="prepStage"
              :form="commonForm"
              :tag-form="tagForm"
              :tag-thresholds="tagThresholds"
              :caption-form="captionForm"
              :clean-form="cleanForm"
              :quality-form="qualityForm"
              :prompt-options="promptOptions"
              :preview-text="previewText"
              :preview-native="previewNative"
            />
            <el-alert
              v-if="reportNotRun"
              type="info"
              :closable="false"
              show-icon
              class="node-drawer__note"
              title="This step has not run yet — the summary above is what it is configured to do."
            />
            <el-alert
              v-else-if="reportError"
              type="error"
              :closable="false"
              show-icon
              class="node-drawer__note"
              :title="reportError"
            />
          </template>

          <template v-else-if="node.type === 'tool'">
            <el-divider content-position="left">Returned value</el-divider>
            <el-text v-if="reportLoading" size="small" type="info">Loading…</el-text>
            <el-alert
              v-else-if="reportNotRun"
              type="info"
              :closable="false"
              show-icon
              title="This step has not run yet."
            />
            <el-alert
              v-else-if="reportError"
              type="error"
              :closable="false"
              show-icon
              :title="reportError"
            />
            <div v-else-if="reportLoaded" class="node-drawer__result">
              <div class="node-drawer__result-bar">result.json</div>
              <pre class="node-drawer__result-body">{{ formattedReport }}</pre>
            </div>
          </template>
        </el-tab-pane>

        <!-- Logs --------------------------------------------------------------- -->
        <el-tab-pane label="Logs" name="logs">
          <div class="node-drawer__log-bar">
            <el-text size="small" type="info">{{ logLabel }}</el-text>
            <el-button v-if="isLive" size="small" type="danger" plain :loading="stopping" @click="stop">
              Stop
            </el-button>
          </div>
          <pre class="node-drawer__log">{{ logText || "(no output yet)" }}</pre>
          <el-text v-if="logError" size="small" type="danger">{{ logError }}</el-text>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-drawer>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { PropType } from "vue";
import { CaretRight, Close, MoreFilled } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";

import NodeRuntimeFields from "./nodeforms/NodeRuntimeFields.vue";
import FolderNodeForm from "./nodeforms/FolderNodeForm.vue";
import ToolNodeForm from "./nodeforms/ToolNodeForm.vue";
import TrainNodeForm from "./nodeforms/TrainNodeForm.vue";

import PrepCommonFields from "../prep/PrepCommonFields.vue";
import TagStageForm from "../prep/TagStageForm.vue";
import CaptionStageForm from "../prep/CaptionStageForm.vue";
import CleanStageForm from "../prep/CleanStageForm.vue";
import QualityStageForm from "../prep/QualityStageForm.vue";
import IndexStageForm from "../prep/IndexStageForm.vue";
import PrepJobSummaryPanel from "../PrepJobSummaryPanel.vue";
import PathValidationFeedback from "../PathValidationFeedback.vue";

import { api } from "../../api";
import { ariaLabel } from "../../lib/aria";
import { formatError } from "../../lib/formatError";
import { ordinalGlyph, ordinals } from "../../lib/workflowGraph";
import {
  consumesInput,
  describeOutput,
  emitsHandle,
  nodeTypeLabel,
} from "../../lib/workflowNodeTypes";
import { resolveText } from "../../lib/workflowVars";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCleanForm,
  defaultCommonForm,
  defaultIndexForm,
  defaultQualityForm,
  defaultTagForm,
  type ModelThresholds,
} from "../../lib/prepStageConfig";
import { useBreakpoint } from "../../composables/useBreakpoint";
import { useDatasetFolderStats } from "../../composables/useDatasetFolderStats";
import type {
  PrepCaptionConfig,
  PrepCleanConfig,
  PrepIndexConfig,
  PrepModelInfo,
  PrepPromptOptions,
  PrepQualityConfig,
  PrepStage,
  PrepTagConfig,
  RunProgress,
} from "../../types/api";
import type {
  DatasetHandle,
  NodeStatus,
  WorkflowGraph,
  WorkflowNode,
  WorkflowState,
} from "../../types/workflow";

const props = defineProps({
  open: { type: Boolean, default: false },
  node: { type: Object as PropType<WorkflowNode | null>, default: null },
  graph: { type: Object as PropType<WorkflowGraph>, required: true },
  state: { type: Object as PropType<WorkflowState>, default: () => ({}) },
  stale: { type: Object as PropType<Record<string, boolean>>, default: () => ({}) },
  workflowId: { type: [Number, String] as PropType<number | string>, required: true },
  /**
   * The runner owns the workflow: every control is disabled and nothing is pushed upward.
   *
   * This is the drawer's half of the editor's contract — the page already refuses the write in
   * `useWorkflowEditor.mutate`, and refusing it *visibly*, here, is the difference between the
   * user being told "Stop to edit" and losing a keystroke to a control that looked live.
   */
  readOnly: { type: Boolean, default: false },
});

const emit = defineEmits<{
  (e: "update:open", open: boolean): void;
  (e: "update:node", node: WorkflowNode): void;
  (e: "run-node", nodeId: string): void;
  (e: "run-from", nodeId: string): void;
}>();

const { isMobile } = useBreakpoint();
const tab = ref("configure");

function close(): void {
  emit("update:open", false);
}

function patchConfig(config: Record<string, unknown>): void {
  if (!props.node || props.readOnly) return;
  emit("update:node", { ...props.node, config });
}

// ------------------------------------------------------------------ node identity and status

const ordinal = computed(() => (props.node ? (ordinals(props.graph)[props.node.id] ?? 0) : 0));
const nodeState = computed(() => (props.node ? props.state.nodes?.[props.node.id] : undefined));
const status = computed<NodeStatus>(() => nodeState.value?.status ?? "pending");
const isStale = computed(() => !!(props.node && props.stale[props.node.id]));
const isLive = computed(() =>
  ["launching", "running", "stopping"].includes(status.value),
);
const consumes = computed(() => (props.node ? consumesInput(props.node.type) : false));
const emits = computed(() => (props.node ? emitsHandle(props.node.type) : false));

/** The status table from the spec's "Node status" section, plus the `disabled` dimming. */
const STATUS_CHIPS: Record<
  NodeStatus,
  { label: string; type: "info" | "success" | "warning" | "danger" | "primary" }
> = {
  pending: { label: "Not run", type: "info" },
  waiting_gpu: { label: "Waiting for GPU", type: "warning" },
  launching: { label: "Starting", type: "primary" },
  running: { label: "Running", type: "primary" },
  stopping: { label: "Stopping", type: "warning" },
  done: { label: "Done", type: "success" },
  failed: { label: "Failed", type: "danger" },
  stopped: { label: "Stopped", type: "warning" },
  skipped: { label: "Skipped", type: "info" },
};

const statusChip = computed(() => {
  if (props.node && !props.node.enabled) {
    return { label: "Disabled", type: "info" as const, effect: "plain" as const };
  }
  const chip = STATUS_CHIPS[status.value] ?? STATUS_CHIPS.pending;
  return { ...chip, effect: isLive.value ? ("dark" as const) : ("light" as const) };
});

// ------------------------------------------------------------------ handles

const DEFAULT_HANDLE = { caption_format: "sidecar", caption_ext: ".txt" };

function inherit(
  path: string,
  input: DatasetHandle | null,
  overrides: Record<string, unknown> = {},
): DatasetHandle {
  return {
    path,
    caption_format: String(
      overrides.caption_format ?? input?.caption_format ?? DEFAULT_HANDLE.caption_format,
    ),
    caption_ext: String(overrides.caption_ext ?? input?.caption_ext ?? DEFAULT_HANDLE.caption_ext),
  };
}

/**
 * What a node *would* emit, with no report to read — the client-side twin of
 * `workflow_graph.effective_output(node, input, report=None)`.
 *
 * This is a prediction and is labelled as one in the UI. The server stays the source of truth:
 * once a node has run, its recorded `output` replaces whatever this returned.
 */
function predictOutput(node: WorkflowNode, input: DatasetHandle | null): DatasetHandle | null {
  const config = node.config ?? {};
  const resolve = (value: unknown): string =>
    typeof value === "string" ? resolveText(value, props.graph.variables) : "";

  switch (node.type) {
    case "folder":
      return inherit(resolve(config.path), null, config);
    case "prep.clean": {
      if (config.in_place) return input;
      const explicit = resolve(config.output_dir).trim();
      if (explicit) return inherit(explicit, input);
      if (!input) return null;
      return inherit(`${input.path.replace(/[\\/]+$/, "")}/cleaned`, input);
    }
    case "prep.tag":
    case "prep.caption":
    case "prep.quality":
    case "prep.index":
      // `prep.quality`'s output_dir is the QUARANTINE folder, not the result: the surviving
      // dataset is still the input folder. Reading it here captions the reject pile.
      return input;
    case "tool":
      // A tool that returns nothing passes its input through, which is the only outcome that can
      // be predicted without running it.
      return input;
    case "train":
      return null;
    default:
      return null;
  }
}

/** Node id -> the handle it emits, and whether that handle is a fact or a prediction. */
const handles = computed(() => {
  const out: Record<string, { handle: DatasetHandle | null; saved: boolean }> = {};
  for (const node of props.graph.nodes) {
    const saved = props.state.nodes?.[node.id]?.output;
    if (saved) {
      out[node.id] = { handle: saved, saved: true };
      continue;
    }
    const input = node.from ? (out[node.from]?.handle ?? null) : null;
    out[node.id] = { handle: predictOutput(node, input), saved: false };
  }
  return out;
});

/** The `From` select's subtitles: what folder each candidate source actually hands over. */
const sourcePaths = computed(() => {
  const out: Record<string, string> = {};
  for (const [id, entry] of Object.entries(handles.value)) {
    if (entry.handle) out[id] = entry.handle.path;
  }
  return out;
});

const sourceNode = computed(() =>
  props.node?.from ? props.graph.nodes.find((n) => n.id === props.node?.from) : undefined,
);
const sourceOrdinal = computed(() =>
  sourceNode.value ? (ordinals(props.graph)[sourceNode.value.id] ?? 0) : 0,
);
const isJump = computed(() => sourceOrdinal.value > 0 && ordinal.value - sourceOrdinal.value > 1);

const sourceEntry = computed(() =>
  props.node?.from ? handles.value[props.node.from] : undefined,
);
const inputHandle = computed<DatasetHandle | null>(() => sourceEntry.value?.handle ?? null);
const inputIsSaved = computed(() => !!sourceEntry.value?.saved);

const outputEntry = computed(() => (props.node ? handles.value[props.node.id] : undefined));
const outputHandle = computed<DatasetHandle | null>(() => outputEntry.value?.handle ?? null);
const outputIsSaved = computed(() => !!outputEntry.value?.saved);

const captionLayoutLabel = computed(() => {
  const handle = inputHandle.value;
  if (!handle) return "—";
  return handle.caption_format === "json"
    ? "captions.json (single index file)"
    : `sidecar files (${handle.caption_ext || ".txt"})`;
});

/**
 * The step's saved input no longer matches what it would consume now — the case a config-only
 * staleness check cannot see, and the reason `saved_input` is stored at all.
 */
const driftedInput = computed(() => {
  const saved = nodeState.value?.saved_input;
  if (!saved || !inputHandle.value) return "";
  return saved.path === inputHandle.value.path ? "" : saved.path;
});

const queuedJobId = computed(() => {
  const result = nodeState.value?.result;
  if (!result || typeof result !== "object") return null;
  const jobId = (result as Record<string, unknown>).job_id;
  return typeof jobId === "number" || typeof jobId === "string" ? jobId : null;
});

// ------------------------------------------------------------------ live folder stats (Input)

const {
  loading: statsLoading,
  error: statsError,
  stats: folderStats,
  load: loadStats,
  clear: clearStats,
} = useDatasetFolderStats();

watch(
  () => [props.open, tab.value, inputHandle.value?.path] as const,
  ([open, active, path]) => {
    if (!open || active !== "input") return;
    if (!path) {
      clearStats();
      return;
    }
    void loadStats(path);
  },
  { immediate: true },
);

const mediaSummary = computed(() => {
  const data = folderStats.value;
  if (!data) return "";
  const images = data.image_count_display ?? String(data.image_count ?? 0);
  const parts = [`${images} images`];
  if (data.video_count) parts.push(`${data.video_count} videos`);
  if (data.has_captions_json) parts.push("captions.json");
  else if (data.caption_txt_files) parts.push(`${data.caption_txt_files} caption files`);
  return parts.join(" · ");
});

// ------------------------------------------------------------------ prep stage forms

const PREP_STAGES: Record<string, PrepStage> = {
  "prep.tag": "tag",
  "prep.caption": "caption",
  "prep.clean": "clean",
  "prep.quality": "quality",
  "prep.index": "index",
};

const prepStage = computed<PrepStage | null>(() =>
  props.node ? (PREP_STAGES[props.node.type] ?? null) : null,
);

// ------------------------------------------------------------------ node report (Output tab)

/** `report.json` for a prep stage, `result.json` for a tool; neither exists for `folder`/`train`. */
const reportSupported = computed(() => !!prepStage.value || props.node?.type === "tool");

const reportLoading = ref(false);
const reportLoaded = ref(false);
const reportNotRun = ref(false);
const reportError = ref("");
const reportData = ref<unknown>(null);

const formattedReport = computed(() => JSON.stringify(reportData.value, null, 2));

let reportGeneration = 0;

async function loadReport(generation: number, workflowId: number | string, nodeId: string): Promise<void> {
  reportLoading.value = true;
  try {
    const result = await api.workflowNodeReport(workflowId, nodeId);
    if (generation !== reportGeneration) return;
    reportData.value = result.report;
    reportLoaded.value = true;
  } catch (e) {
    if (generation !== reportGeneration) return;
    // The route's three 404s are distinguished by their text, not their status: "has not written"
    // is merely early (the step has not run yet), everything else — a corrupt report, or a type
    // this client did not already filter out via `reportSupported` — is a real problem.
    const message = formatError(e);
    if (message.includes("has not written")) {
      reportNotRun.value = true;
    } else {
      reportError.value = message;
    }
  } finally {
    if (generation === reportGeneration) reportLoading.value = false;
  }
}

/**
 * Fetch once per node on the Output tab, re-fetching on every status change so a report written
 * while the user is looking at the tab (the node finishes running) shows up without a reopen.
 */
watch(
  () => [props.open, tab.value, props.node?.id, status.value] as const,
  ([open, active, nodeId]) => {
    reportGeneration += 1;
    reportLoading.value = false;
    reportLoaded.value = false;
    reportNotRun.value = false;
    reportError.value = "";
    reportData.value = null;
    if (!open || active !== "output" || !nodeId || !reportSupported.value) return;
    void loadReport(reportGeneration, props.workflowId, nodeId);
  },
  { immediate: true },
);

/**
 * The stage forms, one object each, handed to the matching component as its v-model.
 *
 * `ref`, not `reactive`: `v-model` on a `const reactive(...)` binding cannot compile a setter, so
 * the compiler warns on every build and quietly drops any whole-object write a child makes. The
 * objects are still mutated in place below — a `ref`'s value is deeply reactive all the same.
 */
const commonForm = ref(defaultCommonForm());
const tagForm = ref(defaultTagForm());
const tagThresholds = ref<Record<string, ModelThresholds>>({});
const tagModels = ref<PrepModelInfo[]>([]);
const captionForm = ref(defaultCaptionForm());
const cleanForm = ref(defaultCleanForm());
const qualityForm = ref(defaultQualityForm());
const indexForm = ref(defaultIndexForm());
const promptOptions = ref<PrepPromptOptions | null>(null);
const previewText = ref("");
const previewNative = ref(false);

/**
 * The node's config, handed to the stage form as its `seed`. The forms copy only the keys they
 * know, so a config written by a newer app version degrades instead of breaking — the same
 * tolerance `parse_prep_config` applies server-side.
 */
const seedSection = computed(() => (props.node?.config ?? null) as Record<string, unknown> | null);

/**
 * Re-seed on every node change. The `:key` on each stage form remounts it so its one-shot
 * `applySeed` runs again; the common form is filled from the *incoming handle*, since in a
 * workflow those three fields come from the edge and not from the node.
 */
watch(
  () => props.node?.id,
  () => {
    Object.assign(commonForm.value, defaultCommonForm());
    Object.assign(tagForm.value, defaultTagForm());
    tagThresholds.value = {};
    Object.assign(captionForm.value, defaultCaptionForm());
    Object.assign(cleanForm.value, defaultCleanForm());
    Object.assign(qualityForm.value, defaultQualityForm());
    Object.assign(indexForm.value, defaultIndexForm());
    previewText.value = "";
    previewNative.value = false;
    tab.value = "configure";
  },
  { immediate: true },
);

// The handle drives the dataset fields the form no longer shows — the quality preview and the
// summary panel still need a path, and it must be the one the edge actually supplies.
watch(
  inputHandle,
  (handle) => {
    commonForm.value.path = handle?.path ?? "";
    commonForm.value.caption_format = handle?.caption_format === "json" ? "json" : "sidecar";
    commonForm.value.caption_ext = handle?.caption_ext || ".txt";
  },
  { immediate: true },
);

/** Order-insensitive structural comparison; `JSON.stringify` alone would trip on key order. */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b || a === null || b === null) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((item, index) => deepEqual(item, b[index]));
  }
  if (typeof a !== "object") return false;
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  return [...keys].every((key) => deepEqual(left[key], right[key]));
}

/** The stage section `buildStageConfig` produces, minus the three fields the edge injects. */
const builtConfig = computed<Record<string, unknown> | null>(() => {
  const stage = prepStage.value;
  if (!stage) return null;
  const payload = buildStageConfig(stage, {
    form: commonForm.value,
    tagForm: tagForm.value,
    tagThresholds: tagThresholds.value,
    tagModels: tagModels.value,
    captionForm: captionForm.value,
    cleanForm: cleanForm.value,
    qualityForm: qualityForm.value,
    indexForm: indexForm.value,
  }) as unknown as Record<string, unknown>;
  return (payload[stage] as Record<string, unknown>) ?? null;
});

/**
 * Push the built config up whenever it stops matching what the node carries.
 *
 * The comparison is what makes this safe to run on every form tick: seeding a saved node
 * reproduces its own config, which compares equal and emits nothing. A node created by
 * `createNode` already carries those same defaults, so opening one is silent too — the only
 * writes left are the registry preselect filling a genuine gap, and the user's own edits.
 */
watch(builtConfig, (config) => {
  if (!config || !props.node || !prepStage.value || props.readOnly) return;
  if (deepEqual(config, props.node.config)) return;
  emit("update:node", { ...props.node, config });
});

// ------------------------------------------------------------------ log tail + progress

const logText = ref("");
const logError = ref("");
const progress = ref<RunProgress | null>(null);
const stopping = ref(false);

let logOffset = 0;
let logTimer: ReturnType<typeof setTimeout> | null = null;
let logGeneration = 0;

function stopPolling(): void {
  if (logTimer) clearTimeout(logTimer);
  logTimer = null;
}

async function pollLog(generation: number): Promise<void> {
  if (!props.node || generation !== logGeneration) return;
  try {
    const result = await api.workflowNodeLog(props.workflowId, props.node.id, logOffset);
    if (generation !== logGeneration) return;
    if (result.chunk) logText.value += result.chunk;
    logOffset = result.offset;
    progress.value = result.progress;
    logError.value = "";
  } catch (e) {
    if (generation !== logGeneration) return;
    // A node that has never run has no log file; that is not an error worth shouting about.
    logError.value = logText.value ? formatError(e) : "";
  }
  if (generation !== logGeneration) return;
  if (isLive.value) logTimer = setTimeout(() => void pollLog(generation), 1500);
}

/**
 * One poller feeds both the Logs tab and the header's progress bar, which is why it runs whenever
 * the drawer is open rather than only while the Logs tab is selected.
 */
watch(
  () => [props.open, props.node?.id, isLive.value] as const,
  ([open, nodeId]) => {
    stopPolling();
    logGeneration += 1;
    if (!open || !nodeId) return;
    logText.value = "";
    logOffset = 0;
    progress.value = null;
    void pollLog(logGeneration);
  },
  { immediate: true },
);

const percent = computed(() => {
  const value = progress.value?.percent;
  if (value == null) return status.value === "done" ? 100 : null;
  return Math.max(0, Math.min(100, Math.round(value)));
});

const progressStatus = computed(() => {
  if (status.value === "failed") return "exception";
  if (status.value === "done") return "success";
  return undefined;
});

const progressLabel = computed(() => {
  const live = progress.value;
  if (live) {
    const parts: string[] = [];
    if (live.phase) parts.push(live.phase);
    if (live.step != null) {
      parts.push(live.max_steps != null ? `${live.step}/${live.max_steps}` : String(live.step));
    }
    if (live.detail) parts.push(String(live.detail));
    if (parts.length) return parts.join(" · ");
  }
  if (status.value === "waiting_gpu") return "Waiting for the GPU to come free.";
  if (isLive.value) return "Waiting for progress…";
  if (nodeState.value?.finished_at) return `Finished ${nodeState.value.finished_at}`;
  return "This step has not run yet.";
});

const logLabel = computed(() =>
  isLive.value ? "Live tail" : "Last run's output",
);

async function stop(): Promise<void> {
  stopping.value = true;
  try {
    await api.cancelWorkflow(props.workflowId);
    ElMessage.info("Stop requested");
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    stopping.value = false;
  }
}

// ------------------------------------------------------------------ header menu

async function onCommand(command: string): Promise<void> {
  const node = props.node;
  // The menu items are disabled while the runner owns the workflow; this is the belt to that
  // braces, so a keyboard-driven command cannot slip past the disabled state either.
  if (!node || props.readOnly) return;
  if (command === "run-node") {
    emit("run-node", node.id);
    return;
  }
  if (command === "run-from") {
    emit("run-from", node.id);
    return;
  }
  if (command === "toggle-enabled") {
    emit("update:node", { ...node, enabled: !node.enabled });
    return;
  }
  if (command === "rename") {
    try {
      const { value } = await ElMessageBox.prompt("Step name", "Rename step", {
        inputValue: node.title,
        inputPattern: /\S/,
        inputErrorMessage: "A step needs a name",
      });
      emit("update:node", { ...node, title: String(value).trim() });
    } catch {
      // dismissed
    }
  }
}
</script>

<style scoped>
.node-drawer :deep(.el-drawer__body) {
  padding: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.node-drawer__body {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}
.node-drawer__head {
  padding: var(--rf-space-md) var(--rf-space-md) var(--rf-space-sm);
  border-bottom: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color);
}
.node-drawer__title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.node-drawer__ordinal {
  font-size: 20px;
  line-height: 1;
  color: var(--el-text-color-secondary);
}
.node-drawer__title {
  display: flex;
  flex-direction: column;
  min-width: 0;
  margin-right: auto;
}
.node-drawer__name {
  font-weight: 600;
  overflow-wrap: anywhere;
}
.node-drawer__type {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.node-drawer__readonly {
  margin: var(--rf-space-sm) var(--rf-space-md) 0;
  width: auto;
}
.node-drawer__progress {
  margin-top: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.node-drawer__progress-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 24px;
}
.node-drawer__tabs {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.node-drawer__tabs :deep(.el-tabs__header) {
  margin: 0;
  padding: 0 var(--rf-space-md);
}
.node-drawer__tabs :deep(.el-tabs__content) {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: var(--rf-space-md);
}
.node-drawer__facts {
  margin: 0 0 var(--rf-space-md);
  display: flex;
  flex-direction: column;
  gap: var(--rf-space-sm);
}
.node-drawer__fact {
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 8px;
  align-items: baseline;
}
.node-drawer__fact dt {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.node-drawer__fact dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.node-drawer__path {
  font-family: var(--rf-font-mono);
  font-size: 12px;
}
.node-drawer__note {
  margin-bottom: var(--rf-space-sm);
}
.node-drawer__stats {
  min-height: 20px;
}
.node-drawer__queued {
  margin-bottom: var(--rf-space-md);
}
.node-drawer__link {
  font-weight: 600;
  color: var(--el-color-primary);
  text-decoration: none;
}
.node-drawer__link:hover {
  text-decoration: underline;
}
.node-drawer__hint {
  display: block;
  margin-top: 4px;
}
.node-drawer__result {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  overflow: hidden;
  margin-bottom: var(--rf-space-md);
}
.node-drawer__result-bar {
  padding: 6px 10px;
  background: var(--el-fill-color-light);
  border-bottom: 1px solid var(--el-border-color-lighter);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--el-text-color-secondary);
}
.node-drawer__result-body {
  margin: 0;
  padding: var(--rf-space-sm);
  font-family: var(--rf-font-mono);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 240px;
  overflow: auto;
}
.node-drawer__log-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}
.node-drawer__log {
  margin: 0;
  padding: var(--rf-space-sm);
  background: var(--el-fill-color-darker, #1a1a1a);
  border-radius: var(--el-border-radius-base);
  font-family: var(--rf-font-mono);
  font-size: 11px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 60vh;
  overflow: auto;
  color: var(--el-text-color-primary);
}
.ml-8 {
  margin-left: 8px;
}
</style>

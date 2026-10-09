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
            :tool-io="toolIo"
            :disabled="readOnly"
            @update:model-value="emit('update:node', $event)"
            @repoint="(sourceId) => emit('repoint', node!.id, sourceId)"
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

            <!--
              Where this step keeps its captions. By default it is the edge's layout ("Same as
              input", with the step that sets it named and a way to it); choosing sidecar or
              captions.json here makes the step convert the folder first and hand the new layout on
              (`workflow_nodes._prep_payload`), which is also the only place a chain that starts with
              a tool - no source folder - can choose.
            -->
            <div v-if="showsCaptionLayout" class="node-drawer__layout">
              <CaptionOutputFormat
                v-model="outputForm"
                :inherited-label="upcomingLayoutLabel"
                :origin-text="captionLayoutOriginText"
                :show-change-link="originHasFormatControl"
                :disabled="readOnly"
                @open-origin="captionLayoutOrigin && emit('open-node', captionLayoutOrigin.id)"
              />
            </div>

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
              :suggestion="tagLineSuggestion"
              :disabled="readOnly"
            />
            <!--
              The control folder is the dataset's, not the step's: while the edge carries one the
              form shows it locked and names the step that sets it (the executor uses it over the
              step's own field, `workflow_graph.edit_control_path`).
            -->
            <EditCaptionStageForm
              v-else-if="prepStage === 'edit_caption'"
              :key="`edit_caption-${node.id}`"
              v-model="editCaptionForm"
              v-model:preview-text="previewText"
              :seed="seedSection as PrepEditCaptionConfig | null"
              :locked-control-path="inputHandle?.control_path ?? ''"
              :control-path-source="controlPathSource"
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
                    <code v-if="inputHandle || !inputRuntimeNote" class="node-drawer__path">{{ inputHandle?.path || "—" }}</code>
                    <el-text v-else size="small" type="warning">{{ inputRuntimeNote }}</el-text>
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
                <div v-if="inputHandle?.control_path" class="node-drawer__fact">
                  <dt>Controls</dt>
                  <dd>
                    <code class="node-drawer__path">{{ inputHandle.control_path }}</code>
                    <el-text size="small" type="info" class="ml-8">from {{ controlPathSource }}</el-text>
                  </dd>
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
              <dd>{{ describeOutput(node, toolIoOf(node, toolIo)) }}</dd>
            </div>
            <div v-if="emits" class="node-drawer__fact">
              <dt>Emits</dt>
              <dd>
                <code v-if="outputHandle || !outputRuntimeNote" class="node-drawer__path">{{ outputHandle?.path || "—" }}</code>
                <el-text v-else size="small" type="warning">{{ outputRuntimeNote }}</el-text>
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

          <!-- Prep writes in place: the emitted folder is the one whose captions this step wrote. -->
          <el-card v-if="captionReviewHref" shadow="never" class="node-drawer__queued">
            <a :href="captionReviewHref" target="_blank" rel="noopener" class="node-drawer__link">
              Review captions &rarr;
            </a>
            <el-text size="small" type="info" class="node-drawer__hint">
              Opens the caption editor on this folder in a new tab, controls beside each target.
            </el-text>
          </el-card>

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
              :edit-caption-form="summaryEditCaptionForm"
              :clean-form="cleanForm"
              :quality-form="qualityForm"
              :prompt-options="promptOptions"
              :preview-text="previewText"
              :preview-native="previewNative"
              :caption-format-label="outputFormatLabel(outputConfigExtras)"
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
import { computed, nextTick, ref, watch } from "vue";
import type { PropType } from "vue";
import { CaretRight, Close, MoreFilled } from "@element-plus/icons-vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { useRouter } from "vue-router";

import NodeRuntimeFields from "./nodeforms/NodeRuntimeFields.vue";
import FolderNodeForm from "./nodeforms/FolderNodeForm.vue";
import ToolNodeForm from "./nodeforms/ToolNodeForm.vue";
import TrainNodeForm from "./nodeforms/TrainNodeForm.vue";
import CaptionOutputFormat, { type CaptionOutputValue } from "./nodeforms/CaptionOutputFormat.vue";

import PrepCommonFields from "../prep/PrepCommonFields.vue";
import TagStageForm from "../prep/TagStageForm.vue";
import CaptionStageForm from "../prep/CaptionStageForm.vue";
import EditCaptionStageForm from "../prep/EditCaptionStageForm.vue";
import CleanStageForm from "../prep/CleanStageForm.vue";
import QualityStageForm from "../prep/QualityStageForm.vue";
import IndexStageForm from "../prep/IndexStageForm.vue";
import PrepJobSummaryPanel from "../PrepJobSummaryPanel.vue";
import PathValidationFeedback from "../PathValidationFeedback.vue";

import { api } from "../../api";
import { ariaLabel } from "../../lib/aria";
import { captionEditorLocation } from "../../lib/captionEditor";
import { formatError } from "../../lib/formatError";
import { ordinalGlyph, ordinals } from "../../lib/workflowGraph";
import {
  consumesInput,
  describeOutput,
  emitsHandle,
  nodeTypeLabel,
  outputFormatChoice,
  outputFormatLabel,
  toolIoOf,
  type ToolIoMap,
} from "../../lib/workflowNodeTypes";
import { resolveText } from "../../lib/workflowVars";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCleanForm,
  defaultCommonForm,
  defaultEditCaptionForm,
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
  PrepEditCaptionConfig,
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
  /** What each Toolbox tool declares it takes and gives (`tool_id` -> io). */
  toolIo: { type: Object as PropType<ToolIoMap>, default: () => ({}) },
  readOnly: { type: Boolean, default: false },
});

const emit = defineEmits<{
  (e: "update:open", open: boolean): void;
  (e: "update:node", node: WorkflowNode): void;
  (e: "run-node", nodeId: string): void;
  (e: "run-from", nodeId: string): void;
  /** Open another step's drawer — the source folder, to change the caption layout there. */
  (e: "open-node", nodeId: string): void;
  /** Connect a step to one further down: the graph reorders, so the page applies it. */
  (e: "repoint", nodeId: string, sourceId: string | null): void;
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

/** A tool that declares it returns a folder: the folder is decided when it runs, not before. */
function decidedAtRunTime(node: WorkflowNode): boolean {
  return node.type === "tool" && toolIoOf(node, props.toolIo)?.output === "folder";
}

/** A tool that declares it hands nothing on emits no handle (`effective_output(tool_output="none")`). */
function emitsNothing(node: WorkflowNode): boolean {
  return node.type === "tool" && toolIoOf(node, props.toolIo)?.output === "none";
}

const RUNTIME_FOLDER_NOTE = "a new folder, decided when this step runs";

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
    // An edit dataset's controls travel with the targets (`workflow_graph._inherit`).
    control_path: String(overrides.control_path ?? input?.control_path ?? "").trim(),
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
      return inherit(resolve(config.path), null, { ...config, control_path: resolve(config.control_path) });
    case "prep.clean": {
      if (config.in_place) return input;
      const explicit = resolve(config.output_dir).trim();
      if (explicit) return inherit(explicit, input);
      if (!input) return null;
      return inherit(`${input.path.replace(/[\\/]+$/, "")}/cleaned`, input);
    }
    case "prep.tag":
    case "prep.caption":
    case "prep.edit_caption": {
      // The step's own caption layout, when it picks one (`workflow_graph.caption_output_override`).
      if (!input) return input;
      const choice = outputFormatChoice(config);
      if (choice === "json") return { ...input, caption_format: "json" };
      if (choice === "sidecar") {
        const ext = String(config.output_ext ?? "").trim() || ".txt";
        return { ...input, caption_format: "sidecar", caption_ext: ext.startsWith(".") ? ext : `.${ext}` };
      }
      return input;
    }
    case "prep.quality":
    case "prep.index":
      // `prep.quality`'s output_dir is the QUARANTINE folder, not the result: the surviving
      // dataset is still the input folder. Reading it here captions the reject pile.
      return input;
    case "tool":
      // Undeclared: a tool that returns nothing passes its input through, the only outcome that
      // can be predicted without running it. A tool declaring it returns a folder has no
      // predictable one (decided when it runs); one declaring `none` emits no handle at all.
      return decidedAtRunTime(node) || emitsNothing(node) ? null : input;
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
  // A folder-declaring tool that has not run: say so instead of showing nothing (or a fake path).
  for (const candidate of props.graph.nodes) {
    if (!out[candidate.id] && decidedAtRunTime(candidate)) out[candidate.id] = RUNTIME_FOLDER_NOTE;
  }
  return out;
});

/** Input tab: the source emits a folder only once it has run. */
const inputRuntimeNote = computed(() =>
  sourceNode.value && decidedAtRunTime(sourceNode.value) && !inputHandle.value
    ? `${RUNTIME_FOLDER_NOTE.replace("this step", sourceNode.value.title)}`
    : "",
);

/** Output tab: this step is a folder-declaring tool that has not run. */
const outputRuntimeNote = computed(() =>
  props.node && decidedAtRunTime(props.node) && !outputHandle.value ? RUNTIME_FOLDER_NOTE : "",
);

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

/**
 * The step that sets the incoming control folder: the nearest ancestor whose own input does not
 * already carry it (a source folder, or a tool that returned one). "① Source folder".
 */
const controlPathSource = computed(() => {
  const control = inputHandle.value?.control_path;
  if (!control) return "";
  const byId = new Map(props.graph.nodes.map((n) => [n.id, n]));
  let origin = sourceNode.value;
  while (origin?.from && handles.value[origin.from]?.handle?.control_path === control) {
    origin = byId.get(origin.from);
  }
  if (!origin) return "";
  return `${ordinalGlyph(ordinals(props.graph)[origin.id] ?? 0)} ${origin.title}`;
});

/** Steps that leave the folder (and its caption lines) as they got it, so a tag step above shows through. */
const TAG_PASS_THROUGH = ["prep.caption", "prep.edit_caption", "prep.quality", "prep.index"];

/**
 * For a caption step: the nearest tag step above it (through pass-through steps) and the line it
 * writes tags to. Only ever shown as a hint — the caption form's grounding line is never rewritten
 * on its own, so opening a step stays free of edits.
 */
const tagLineSuggestion = computed(() => {
  if (props.node?.type !== "prep.caption") return null;
  const byId = new Map(props.graph.nodes.map((n) => [n.id, n]));
  const seen = new Set<string>();
  let up = sourceNode.value;
  while (up && !seen.has(up.id)) {
    seen.add(up.id);
    if (up.type === "prep.tag" && up.enabled) {
      const raw = Number((up.config as Record<string, unknown>).target_line ?? 1);
      const line = Number.isFinite(raw) && raw >= 1 ? Math.trunc(raw) : 1;
      return { line, source: `Tags step ${ordinalGlyph(ordinals(props.graph)[up.id] ?? 0)}` };
    }
    if ((up.enabled && !TAG_PASS_THROUGH.includes(up.type) && up.type !== "prep.tag") || !up.from) return null;
    up = byId.get(up.from);
  }
  return null;
});

/** Stages that read or write captions; cleanup, quality and the index see images only. */
const showsCaptionLayout = computed(
  () => prepStage.value === "tag" || prepStage.value === "caption" || prepStage.value === "edit_caption",
);

/**
 * The handles as the *next* run would see them: a stale step's recorded output is what it emitted
 * last time, not what it will emit now. The caption-layout row uses these, so changing the source
 * folder's format shows up on the caption step right away instead of after the next run.
 */
const upcomingHandles = computed(() => {
  const out: Record<string, DatasetHandle | null> = {};
  for (const node of props.graph.nodes) {
    const saved = props.state.nodes?.[node.id]?.output;
    if (saved && !props.stale[node.id]) {
      out[node.id] = saved;
      continue;
    }
    out[node.id] = predictOutput(node, node.from ? (out[node.from] ?? null) : null);
  }
  return out;
});

const upcomingInput = computed<DatasetHandle | null>(() =>
  props.node?.from ? (upcomingHandles.value[props.node.from] ?? null) : null,
);

const upcomingLayoutLabel = computed(() => {
  const handle = upcomingInput.value;
  if (!handle) return "—";
  return handle.caption_format === "json"
    ? "captions.json (single index file)"
    : `sidecar files (${handle.caption_ext || ".txt"})`;
});

/**
 * The step that sets the incoming caption layout: the nearest ancestor whose own input does not
 * already carry the same format and extension (a source folder, or a tool that returned one).
 */
const captionLayoutOrigin = computed<WorkflowNode | null>(() => {
  const handle = upcomingInput.value;
  if (!handle) return null;
  const sameLayout = (other: DatasetHandle | null | undefined) =>
    !!other && other.caption_format === handle.caption_format && other.caption_ext === handle.caption_ext;
  const byId = new Map(props.graph.nodes.map((n) => [n.id, n]));
  let origin = sourceNode.value;
  while (origin?.from && sameLayout(upcomingHandles.value[origin.from])) {
    origin = byId.get(origin.from);
  }
  return origin ?? null;
});

/** `① Source folder` - the step that sets the incoming layout, for the "set on" hint. */
const captionLayoutOriginText = computed(() => {
  const origin = captionLayoutOrigin.value;
  if (!origin) return "";
  return `${ordinalGlyph(ordinals(props.graph)[origin.id] ?? 0)} ${origin.title}`;
});

/** Only steps with a caption-format control of their own can be sent to from here. */
const originHasFormatControl = computed(() =>
  ["folder", "prep.tag", "prep.caption", "prep.edit_caption"].includes(
    captionLayoutOrigin.value?.type ?? "",
  ),
);

/** The summary panel shows the folder the step will actually use: the edge's, when it has one. */
const summaryEditCaptionForm = computed(() => ({
  ...editCaptionForm.value,
  control_path: inputHandle.value?.control_path || editCaptionForm.value.control_path,
}));

const router = useRouter();

/** Caption editor link once a caption/edit_caption step has actually run (its output is saved). */
const captionReviewHref = computed(() => {
  const handle = outputHandle.value;
  if (!handle?.path || !outputIsSaved.value) return "";
  if (prepStage.value !== "caption" && prepStage.value !== "edit_caption") return "";
  const controlPath =
    prepStage.value === "edit_caption" ? summaryEditCaptionForm.value.control_path : handle.control_path ?? "";
  return router.resolve(
    captionEditorLocation({
      path: handle.path,
      control_path: controlPath,
      format: handle.caption_format === "json" ? "json" : "sidecar",
      ext: handle.caption_ext,
    })
  ).href;
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
  "prep.edit_caption": "edit_caption",
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
const editCaptionForm = ref(defaultEditCaptionForm());
const cleanForm = ref(defaultCleanForm());
const qualityForm = ref(defaultQualityForm());
const indexForm = ref(defaultIndexForm());
const promptOptions = ref<PrepPromptOptions | null>(null);
const previewText = ref("");
const previewNative = ref(false);

/**
 * The step's own caption layout (`output_format` / `output_ext` in its config). Not part of any
 * stage form - it belongs to the node - so it is seeded from the node on every selection and
 * merged into `builtConfig`, which is what keeps the stage forms' whole-config writes from
 * dropping it.
 */
const outputForm = ref<CaptionOutputValue>({ format: "inherit", ext: ".txt" });

function outputFormOf(config: Record<string, unknown> | undefined): CaptionOutputValue {
  const ext = typeof config?.output_ext === "string" && config.output_ext ? config.output_ext : ".txt";
  return { format: outputFormatChoice(config), ext };
}

/** The keys `outputForm` adds to a config: none while it inherits, so an untouched step is unchanged. */
const outputConfigExtras = computed<Record<string, unknown>>(() => {
  if (!showsCaptionLayout.value) return {};
  const { format, ext } = outputForm.value;
  if (format === "inherit") return {};
  return format === "json" ? { output_format: "json" } : { output_format: "sidecar", output_ext: ext || ".txt" };
});

/**
 * The node's config, handed to the stage form as its `seed`. The forms copy only the keys they
 * know, so a config written by a newer app version degrades instead of breaking — the same
 * tolerance `parse_prep_config` applies server-side.
 */
const seedSection = computed(() => (props.node?.config ?? null) as Record<string, unknown> | null);

/**
 * Set while the forms are being reset for a newly selected node. The drawer stays mounted between
 * selections, so the reset below changes `builtConfig` to the *defaults* before the remounted
 * stage form has seeded the node's saved config — and without this the watcher would push those
 * defaults up as an edit, and the editor would autosave them over what the user had saved.
 *
 * Relies on the node-id watcher below being registered before `watch(builtConfig)`: both run in
 * the same pre-flush, in registration order, so the flag is up by the time the reset's
 * `builtConfig` callback runs. Keep that order.
 */
let reseeding = false;

/**
 * The handle drives the dataset fields the form no longer shows — the quality preview and the
 * summary panel still need a path and caption layout, and they must be the ones the edge actually
 * supplies, never the form defaults.
 */
function applyHandle(handle: DatasetHandle | null): void {
  commonForm.value.path = handle?.path ?? "";
  commonForm.value.caption_format = handle?.caption_format === "json" ? "json" : "sidecar";
  commonForm.value.caption_ext = handle?.caption_ext || ".txt";
}

/**
 * Re-seed on every node change. The `:key` on each stage form remounts it so its one-shot
 * `applySeed` runs again; the common form is filled from the *incoming handle*, since in a
 * workflow those three fields come from the edge and not from the node. (Sibling steps share one
 * handle object, so the `inputHandle` watcher alone would not re-fire between them.)
 */
watch(
  () => props.node?.id,
  () => {
    reseeding = true;
    // Cleared after this flush whether or not the reset changed `builtConfig` at all.
    void nextTick(() => {
      reseeding = false;
    });
    Object.assign(commonForm.value, defaultCommonForm());
    applyHandle(inputHandle.value);
    Object.assign(tagForm.value, defaultTagForm());
    tagThresholds.value = {};
    Object.assign(captionForm.value, defaultCaptionForm());
    Object.assign(editCaptionForm.value, defaultEditCaptionForm());
    Object.assign(cleanForm.value, defaultCleanForm());
    Object.assign(qualityForm.value, defaultQualityForm());
    Object.assign(indexForm.value, defaultIndexForm());
    outputForm.value = outputFormOf(props.node?.config);
    previewText.value = "";
    previewNative.value = false;
    tab.value = "configure";
  },
  { immediate: true },
);

watch(inputHandle, applyHandle, { immediate: true });

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
    editCaptionForm: editCaptionForm.value,
    cleanForm: cleanForm.value,
    qualityForm: qualityForm.value,
    indexForm: indexForm.value,
  }) as unknown as Record<string, unknown>;
  const section = (payload[stage] as Record<string, unknown>) ?? null;
  return section ? { ...section, ...outputConfigExtras.value } : null;
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
  if (reseeding) {
    reseeding = false;
    return;
  }
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
.node-drawer__layout {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--rf-space-sm);
  margin-bottom: var(--rf-space-md);
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

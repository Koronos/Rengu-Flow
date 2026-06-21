<template>
  <div>
    <div class="page-head run-detail__head">
      <el-button :icon="ArrowLeft" @click="goBack">Runs</el-button>
      <span class="run-detail__title">{{ title }}</span>
      <div class="run-detail__head-actions">
        <template v-if="runDir">
          <el-button
            v-if="!runIsActive"
            type="primary"
            @click="goContinueTraining"
          >
            Continue training…
          </el-button>
          <el-button v-if="mode === 'job' && job?.id" @click="newRunFromThisConfig">New run from this config</el-button>
          <el-button :loading="tbLoading" @click="openTensorboardForRun">Open TensorBoard</el-button>
          <el-button @click="compareFolder">Compare folder</el-button>
          <el-button
            v-if="tbStatus?.running"
            :loading="tbLoading"
            type="danger"
            plain
            @click="stopTensorboardForRun"
          >
            Stop TensorBoard
          </el-button>
        </template>
        <template v-else-if="mode === 'job' && job?.id">
          <el-button type="primary" @click="goContinueTraining">Retry run</el-button>
          <el-button @click="newRunFromThisConfig">New run from this config</el-button>
        </template>
      </div>
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mt-12" />
    <el-alert
      v-if="streamError"
      type="warning"
      :title="streamError"
      show-icon
      class="mt-12"
    />

    <el-alert
      v-if="diskExportWait"
      type="warning"
      show-icon
      class="mt-12"
      title="Training paused — disk full during model export"
    >
      Free disk space on the run directory, then use <strong>Continue export</strong> below.
      Weights stay loaded on the GPU until the export succeeds or you quit.
      See
      <router-link :to="{ path: '/docs', query: { doc: 'docs/user/checkpoint-and-save.md' } }">
        Checkpoints &amp; export
      </router-link>.
    </el-alert>

    <el-card shadow="never" class="mt-12">
      <el-descriptions :column="isMobile ? 1 : 2" border size="small">
        <el-descriptions-item label="Run dir">
          <el-text class="mono" truncated>{{ runDir || "—" }}</el-text>
        </el-descriptions-item>
        <el-descriptions-item v-if="job" label="State">
          <el-tag :type="runStateTag(job.state)" size="small">
            {{ runStateLabel(job.state) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="job" label="PID">
          {{ job.pid ?? "—" }}
        </el-descriptions-item>
      </el-descriptions>

      <div
        v-if="(tbStatus?.running && tbStatus.url) || (runDir && !runIsActive) || (!runDir && mode === 'job' && job?.id)"
        class="run-detail__hint"
      >
        <span v-if="tbStatus?.running && tbStatus.url" class="run-detail__tb">
          <el-text type="info" size="small">TensorBoard:</el-text>
          <el-link :href="tbStatus.url" target="_blank" type="primary">{{ tbStatus.url }}</el-link>
          <el-text type="info" size="small">(opens in a new tab)</el-text>
        </span>
        <el-text v-if="runDir && !runIsActive" type="info" size="small">
          “Continue training…” loads this run's TOML so you can edit epochs/steps and resume in this folder.
        </el-text>
        <el-text v-else-if="!runDir && mode === 'job' && job?.id" type="info" size="small">
          No checkpoint to resume — “Retry run” re-runs this record from scratch; “New run from this config” creates a separate one.
        </el-text>
      </div>

      <p v-if="signalsAvailable" class="signals-intro">
        <el-text type="info" size="small">{{ signalSectionHint }}</el-text>
        <el-button type="primary" link size="small" @click="signalDocOpen = true">
          Signal files guide
        </el-button>
      </p>
      <RunSignalActions
        :available="signalsAvailable"
        :disk-export-wait="diskExportWait"
        :show-unavailable-hint="true"
        :exclude-groups="['Preview']"
        @send="sendSignal"
      />
      <DocMarkdownDrawer v-model="signalDocOpen" :doc-path="signalDocPath" />
    </el-card>

    <el-card v-if="runIsActive && progress" shadow="never" class="mt-12">
      <template #header>Progress</template>
      <RunProgress
        :progress="progress"
        :preview-steps="previewSteps"
        :checkpoint-steps="checkpointSteps"
      />
    </el-card>

    <el-card v-if="mode === 'job' && job?.id && signalsAvailable" shadow="never" class="mt-12">
      <el-collapse>
        <el-collapse-item title="Live preview settings" name="preview">
          <LivePreviewEditor :job-id="job.id" />
        </el-collapse-item>
      </el-collapse>
    </el-card>

    <el-card shadow="never" class="mt-12">
      <template #header>Loss</template>
      <RunLossMonitor
        :scalars="metrics"
        :preview-images="previewImages"
        :loading="metricsLoading"
        :run-name="title"
        @refresh="fetchMetrics"
      />
    </el-card>

    <el-card v-if="mode === 'job'" shadow="never" class="mt-12">
      <template #header>Log</template>
      <pre ref="logPreRef" class="log-pre" @scroll="onLogScroll">{{ logText || "(waiting for output…)" }}</pre>
    </el-card>

    <el-card v-if="artifacts.length" shadow="never" class="mt-12">
      <template #header>Artifacts</template>
      <el-table :data="artifacts" size="small">
        <el-table-column prop="type" label="Type" width="120" />
        <el-table-column prop="path" label="Path" show-overflow-tooltip />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import { useAutoRefresh } from "../composables/useAutoRefresh";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useJobLogStream } from "../composables/useJobLogStream";
import { useTrainLiveStream } from "../composables/useTrainLiveStream";
import { useTensorboard } from "../composables/useTensorboard";
import { formatError } from "../lib/formatError";
import { runStateTag, runStateLabel } from "../lib/runState";
import RunLossMonitor from "../components/RunLossMonitor.vue";
import type { ScalarPoint } from "../lib/scalarChart";
import type { RunPreviewImageRef, RunProgress as RunProgressData } from "../types/api";
import DocMarkdownDrawer from "../components/DocMarkdownDrawer.vue";
import RunSignalActions from "../components/RunSignalActions.vue";
import LivePreviewEditor from "../components/LivePreviewEditor.vue";
import RunProgress from "../components/RunProgress.vue";
import { SIGNAL_DOC_PATH, SIGNAL_SECTION_HINT } from "../lib/signalHelp";
import { fsRunSignalsAvailable, jobSignalsAvailable } from "../lib/trainingSignals";
import { useConfigEditorStore } from "../stores/configEditor";
import type { FsRunRecord, JobRecord } from "../types/api";

const props = defineProps({
  mode: { type: String, required: true },
  name: { type: String, default: "" },
});

const route = useRoute();
const router = useRouter();
const { isMobile } = useBreakpoint();
const editor = useConfigEditorStore();

const signalSectionHint = SIGNAL_SECTION_HINT;
const signalDocPath = SIGNAL_DOC_PATH;
const signalDocOpen = ref(false);

const job = ref<JobRecord | null>(null);
const fsRun = ref<FsRunRecord | null>(null);
const jobArtifacts = ref<Record<string, unknown>[]>([]);
const metrics = ref<Record<string, ScalarPoint[]>>({});
const previewImages = ref<RunPreviewImageRef[]>([]);
const checkpointSteps = ref<number[]>([]);
const previewSteps = computed(() =>
  previewImages.value.map((p) => p.step).filter((s): s is number => s != null)
);
const metricsLoading = ref(false);
const progress = ref<RunProgressData | null>(null);
const error = ref("");
const outputDir = ref("output");
const { tbLoading, tbStatus, refreshTbStatus, openTensorboard, stopTensorboard } = useTensorboard(
  () => String(job.value?.output_dir || outputDir.value || "output")
);
let cachedPreviewRunDir: string | null = null;

const key = computed(() => (props.mode === "job" ? route.params.id : route.params.name));
const jobId = computed(() => {
  const runKey = key.value;
  return props.mode === "job" && runKey != null ? String(Array.isArray(runKey) ? runKey[0] : runKey) : "";
});
const runIsActive = computed(() => {
  if (props.mode === "job") {
    const s = job.value?.state;
    return s === "running" || s === "stopping";
  }
  return false;
});

// While the run is live, drive progress / metrics / previews / log from the same WebSocket
// stream the Runs page uses, so the detail updates by instant push instead of waiting on the
// HTTP poll. The poll below stays as the fallback for when the socket can't connect.
const liveJobId = computed(() => (props.mode === "job" && runIsActive.value ? jobId.value : ""));
const {
  progress: liveProgress,
  logText: liveLogText,
  streamError: liveStreamError,
  useHttpFallback: liveUseHttpFallback,
} = useTrainLiveStream(liveJobId, {
  onRunFinished: () => {
    void refreshMetricsNow();
  },
});

// Terminal runs (and the brief moment before the socket connects) show their final log over
// the dedicated log stream; the live stream above only runs while the job is active.
const logJobId = computed(() => (props.mode === "job" && !runIsActive.value ? jobId.value : ""));
const { logText: jobLogText, streamError: jobLogError } = useJobLogStream(logJobId);

const logText = computed(() => (runIsActive.value ? liveLogText.value : jobLogText.value));
const streamError = computed(() => (runIsActive.value ? liveStreamError.value : jobLogError.value));

// The live socket drives the progress bar (instant push) while connected; on HTTP fallback the
// poll owns it instead. Loss metrics / previews are deliberately NOT pushed from here — the Loss
// monitor owns its own opt-in refresh (TensorBoard-style), so the charts only move when asked.
watch(liveProgress, (p) => {
  if (!liveUseHttpFallback.value && p) progress.value = p;
});

// Sticky auto-scroll for the log: follow new output while pinned to the bottom, but
// stop following once the user scrolls up (recovered from the old runs-page panel).
const logPreRef = ref<HTMLElement | null>(null);
let userScrolledUp = false;
function onLogScroll(): void {
  const el = logPreRef.value;
  if (!el) return;
  const threshold = 48;
  userScrolledUp = el.scrollTop + el.clientHeight < el.scrollHeight - threshold;
}
async function scrollLogToEnd(): Promise<void> {
  await nextTick();
  const el = logPreRef.value;
  if (!el || userScrolledUp) return;
  el.scrollTop = el.scrollHeight;
}
watch(logText, () => {
  void scrollLogToEnd();
});

function runFolderName(dir: string | null | undefined): string {
  if (!dir) return "";
  const parts = String(dir).split(/[\\/]/).filter(Boolean);
  return parts[parts.length - 1] || "";
}
const title = computed(() => {
  if (props.mode === "job") {
    return (
      job.value?.run_name ||
      runFolderName(job.value?.run_dir) ||
      `Job ${key.value}`
    );
  }
  return fsRun.value?.name || runFolderName(fsRun.value?.path) || `Run ${key.value}`;
});

const runDir = computed(() => job.value?.run_dir || fsRun.value?.path);
const status = computed(() => job.value?.status || fsRun.value?.status || null);
const diskExportWait = computed(() => status.value?.phase === "waiting_disk_export");
const signalsAvailable = computed(() =>
  props.mode === "job" ? jobSignalsAvailable(job.value) : fsRunSignalsAvailable(fsRun.value)
);
const artifacts = computed(() => fsRun.value?.artifacts?.length ? fsRun.value.artifacts : jobArtifacts.value);
function goBack() {
  router.push("/runs");
}

function goContinueTraining() {
  if (props.mode === "job" && job.value?.id) {
    router.push({ name: "run-continue", params: { id: String(job.value.id) } });
    return;
  }
  // Filesystem-only run with no job id: fall back to the path query.
  if (!runDir.value) return;
  router.push({ name: "run-new", query: { continue_run: runDir.value } });
}

/** Retry path for a run with no checkpoint to resume (e.g. failed at setup): open a fresh,
 *  editable run pre-filled with this run's config, ready to fix and add to the queue. */
async function newRunFromThisConfig() {
  if (!job.value?.id) return;
  try {
    const { content } = await api.seedJobConfig(String(job.value.id));
    await editor.fetchSchema();
    await editor.loadContent(content);
    router.push({ name: "run-new" });
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function openTensorboardForRun() {
  try {
    await openTensorboard({ onError: (msg) => { error.value = msg; } });
    await refreshTbStatus();
  } catch {
    /* ElMessage already shown */
  }
}

function compareFolder() {
  // Open the comparison view on this run's output folder (same folder Open TensorBoard uses).
  const dir = String(job.value?.output_dir || outputDir.value || "output");
  router.push({ path: "/compare", query: { output_dir: dir } });
}

async function stopTensorboardForRun() {
  try {
    await stopTensorboard({ onError: (msg) => { error.value = msg; } });
    await refreshTbStatus();
  } catch {
    /* ElMessage already shown */
  }
}

async function loadJobMetadata(runKey: string): Promise<void> {
  if (!job.value?.run_dir) {
    fsRun.value = null;
    jobArtifacts.value = [];
    return;
  }
  if (job.value.run_dir !== cachedPreviewRunDir || job.value.state === "running") {
    cachedPreviewRunDir = job.value.run_dir;
    const [previewResult, artifactsResult] = await Promise.allSettled([
      api.previewJobImport(job.value.run_dir),
      api.jobArtifacts(runKey),
    ]);
    if (previewResult.status === "fulfilled") {
      const preview = previewResult.value as { run?: FsRunRecord };
      fsRun.value = preview.run || null;
    }
    if (artifactsResult.status === "fulfilled") {
      const art = artifactsResult.value as { artifacts?: Record<string, unknown>[] };
      jobArtifacts.value = art.artifacts || [];
    }
  }
}

// Refresh the run record (state / pid / metadata) and, on the HTTP fallback, the progress bar.
// Loss metrics are fetched separately by fetchMetrics() so they stay on their own opt-in cadence.
async function poll(signal: AbortSignal) {
  try {
    if (props.mode === "job") {
      const runKey = jobId.value;
      if (!runKey) return;
      const jobResult = await api.getJob(runKey);
      if (signal.aborted) return;
      job.value = jobResult as JobRecord;
      progress.value =
        (jobResult as JobRecord & { progress?: RunProgressData | null }).progress ?? null;
      await loadJobMetadata(runKey);
    } else {
      const runName = Array.isArray(key.value) ? key.value[0] : key.value;
      if (!runName) return;
      const runResult = await api.getFsRun(String(runName), outputDir.value);
      if (signal.aborted) return;
      fsRun.value = runResult as FsRunRecord;
    }
    if (signal.aborted) return;
    error.value = "";
  } catch (e) {
    if (signal.aborted) return;
    error.value = formatError(e);
  }
}

// Loss metrics + previews are fetched on demand: once on load, then via the Loss monitor's manual
// reload or its opt-in auto-update. Kept off the live progress stream so the charts behave like
// TensorBoard — static unless you ask for updates.
async function fetchMetrics(): Promise<void> {
  const runKey = props.mode === "job" ? jobId.value : key.value;
  const name = Array.isArray(runKey) ? runKey[0] : runKey;
  if (!name) return;
  metricsLoading.value = true;
  try {
    const result =
      props.mode === "job"
        ? await api.jobMetrics(String(name))
        : await api.fsMetrics(String(name), outputDir.value);
    const m = result as {
      scalars?: Record<string, ScalarPoint[]>;
      preview_images?: RunPreviewImageRef[];
    };
    metrics.value = m.scalars || {};
    previewImages.value = m.preview_images || [];
  } catch (e) {
    error.value = formatError(e);
  } finally {
    metricsLoading.value = false;
  }
  // Checkpoint ticks for the progress bar; decorative, so keep its failure off the metrics error banner.
  if (props.mode === "job") {
    try {
      const cp = await api.jobCheckpoints(String(name));
      checkpointSteps.value = cp.checkpoints.map((c) => c.step).filter((s) => s != null);
    } catch (e) {
      console.warn("checkpoint ticks unavailable:", e);
    }
  }
}

// The poll keeps the run record fresh; progress arrives by push (live stream) with this as the
// HTTP fallback, so it only needs to run on a timer while that fallback is in use.
const { refreshNow: refreshMetricsNow } = useAutoRefresh({
  refresh: poll,
  isActive: () => runIsActive.value && liveUseHttpFallback.value,
});

async function sendSignal(type: string) {
  if (!signalsAvailable.value) {
    ElMessage.warning("Signals are only available while training is active.");
    return;
  }
  error.value = "";
  try {
    const runKey = Array.isArray(key.value) ? key.value[0] : key.value;
    if (!runKey) return;
    if (props.mode === "job") {
      await api.sendJobSignal(runKey, type);
    } else {
      await api.fsSignal(runKey, type, outputDir.value);
    }
    ElMessage.success(`Signal "${type}" sent`);
    void refreshMetricsNow();
  } catch (e) {
    const msg = formatError(e);
    error.value = msg;
    ElMessage.error(msg);
  }
}

onMounted(() => {
  refreshTbStatus();
  void fetchMetrics();
});

watch(key, () => {
  cachedPreviewRunDir = null;
  jobArtifacts.value = [];
  job.value = null;
  fsRun.value = null;
  previewImages.value = [];
  checkpointSteps.value = [];
  metrics.value = {};
  void refreshMetricsNow();
  void fetchMetrics();
});
</script>

<style scoped>
.mt-12 {
  margin-top: 12px;
}
.run-detail__head {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: var(--rf-space-sm) 0;
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.run-detail__title {
  font-size: 18px;
  font-weight: 600;
  flex: 1;
  min-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.run-detail__head-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.run-detail__hint {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  margin-top: 12px;
}
.run-detail__tb {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.signals-intro {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 8px;
  margin: 8px 0 0;
}
.mono {
  font-family: ui-monospace, monospace;
  font-size: 12px;
}
.loss-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.log-pre {
  margin: 0;
  max-height: 420px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, monospace;
  font-size: 12px;
  line-height: 1.45;
}
</style>

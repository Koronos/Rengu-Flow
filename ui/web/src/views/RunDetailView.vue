<template>
  <div>
    <el-page-header @back="goBack">
      <template #content>
        <span class="page-title-inline">{{ title }}</span>
      </template>
    </el-page-header>

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
          <el-tag :type="job.state === 'running' ? 'success' : 'info'" size="small">
            {{ job.state }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item v-if="job" label="PID">
          {{ job.pid ?? "—" }}
        </el-descriptions-item>
        <el-descriptions-item v-if="status" label="Status file">
          step {{ status.step }}, loss {{ formatLoss(status.loss) }}
        </el-descriptions-item>
      </el-descriptions>

      <div v-if="runDir" class="continue-row">
        <el-button type="primary" @click="goContinueTraining">
          Continue training…
        </el-button>
        <el-button :loading="tbLoading" @click="openTensorboardForRun">
          Open TensorBoard
        </el-button>
        <el-button
          v-if="tbStatus?.running"
          :loading="tbLoading"
          type="danger"
          plain
          @click="stopTensorboardForRun"
        >
          Stop TensorBoard
        </el-button>
        <el-link
          v-if="tbStatus?.running && tbStatus.url"
          :href="tbStatus.url"
          target="_blank"
          type="primary"
        >
          {{ tbStatus.url }}
        </el-link>
        <el-text type="info" size="small">
          Load run TOML, edit epochs/steps, resume in this folder
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
        @send="sendSignal"
      />
      <DocMarkdownDrawer v-model="signalDocOpen" :doc-path="signalDocPath" />
    </el-card>

    <el-card shadow="never" class="mt-12">
      <template #header>
        <div class="loss-card-head">
          <span>Loss</span>
          <AutoRefreshBar
            :interval-sec="intervalSec"
            :refreshing="metricsRefreshing"
            :polling="metricsPolling"
            :last-updated="metricsLastUpdated"
            :paused="metricsPaused"
            @update:interval-sec="setMetricsInterval"
            @refresh="refreshMetricsNow"
          />
        </div>
      </template>
      <RunLossMonitor
        :scalars="metrics"
        :preview-images="previewImages"
        :loading="metricsLoading"
        :loading-strong="metricsRefreshing"
      />
    </el-card>

    <el-card v-if="mode === 'job'" shadow="never" class="mt-12">
      <template #header>Log</template>
      <pre class="log-pre">{{ logText || "(waiting for output…)" }}</pre>
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
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { api } from "../api";
import AutoRefreshBar from "../components/AutoRefreshBar.vue";
import { useAutoRefresh } from "../composables/useAutoRefresh";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useJobLogStream } from "../composables/useJobLogStream";
import { useTensorboard } from "../composables/useTensorboard";
import { formatError } from "../lib/formatError";
import RunLossMonitor from "../components/RunLossMonitor.vue";
import type { ScalarPoint } from "../lib/scalarChart";
import type { RunPreviewImageRef } from "../types/api";
import DocMarkdownDrawer from "../components/DocMarkdownDrawer.vue";
import RunSignalActions from "../components/RunSignalActions.vue";
import { SIGNAL_DOC_PATH, SIGNAL_SECTION_HINT } from "../lib/signalHelp";
import { fsRunSignalsAvailable, jobSignalsAvailable } from "../lib/trainingSignals";
import type { FsRunRecord, JobRecord } from "../types/api";

const props = defineProps({
  mode: { type: String, required: true },
  name: { type: String, default: "" },
});

const route = useRoute();
const router = useRouter();
const { isMobile } = useBreakpoint();

const signalSectionHint = SIGNAL_SECTION_HINT;
const signalDocPath = SIGNAL_DOC_PATH;
const signalDocOpen = ref(false);

const job = ref<JobRecord | null>(null);
const fsRun = ref<FsRunRecord | null>(null);
const jobArtifacts = ref<Record<string, unknown>[]>([]);
const metrics = ref<Record<string, ScalarPoint[]>>({});
const previewImages = ref<RunPreviewImageRef[]>([]);
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
const { logText, streamError } = useJobLogStream(jobId);

const title = computed(() =>
  props.mode === "job" ? `Job ${key.value}` : `Run ${key.value}`
);

const runDir = computed(() => job.value?.run_dir || fsRun.value?.path);
const status = computed(() => job.value?.status || fsRun.value?.status || null);
const diskExportWait = computed(() => status.value?.phase === "waiting_disk_export");
const signalsAvailable = computed(() =>
  props.mode === "job" ? jobSignalsAvailable(job.value) : fsRunSignalsAvailable(fsRun.value)
);
const artifacts = computed(() => fsRun.value?.artifacts?.length ? fsRun.value.artifacts : jobArtifacts.value);
const runIsActive = computed(() => {
  if (props.mode === "job") {
    const s = job.value?.state;
    return s === "running" || s === "stopping";
  }
  return false;
});

function formatLoss(v: number | null | undefined): string | number | null | undefined {
  return typeof v === "number" ? v.toFixed(6) : v;
}

function goBack() {
  router.push("/runs");
}

function goContinueTraining() {
  if (!runDir.value) return;
  router.push({ name: "configs-new", query: { continue_run: runDir.value } });
}

async function openTensorboardForRun() {
  try {
    await openTensorboard({ onError: (msg) => { error.value = msg; } });
    await refreshTbStatus();
  } catch {
    /* ElMessage already shown */
  }
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

async function poll(signal: AbortSignal) {
  try {
    if (props.mode === "job") {
      const runKey = jobId.value;
      if (!runKey) return;
      const [jobResult, metricsResult] = await Promise.all([
        api.getJob(runKey),
        api.jobMetrics(runKey),
      ]);
      if (signal.aborted) return;
      job.value = jobResult as JobRecord;
      const m = metricsResult as {
        scalars?: Record<string, ScalarPoint[]>;
        preview_images?: RunPreviewImageRef[];
      };
      metrics.value = m.scalars || {};
      previewImages.value = m.preview_images || [];
      await loadJobMetadata(runKey);
    } else {
      const runName = Array.isArray(key.value) ? key.value[0] : key.value;
      if (!runName) return;
      const [runResult, metricsResult] = await Promise.all([
        api.getFsRun(String(runName), outputDir.value),
        api.fsMetrics(String(runName), outputDir.value),
      ]);
      if (signal.aborted) return;
      fsRun.value = runResult as FsRunRecord;
      const m = metricsResult as {
        scalars?: Record<string, ScalarPoint[]>;
        preview_images?: RunPreviewImageRef[];
      };
      metrics.value = m.scalars || {};
      previewImages.value = m.preview_images || [];
    }
    if (signal.aborted) return;
    error.value = "";
  } catch (e) {
    if (signal.aborted) return;
    error.value = formatError(e);
  }
}

const {
  intervalSec,
  isLoading: metricsLoading,
  refreshing: metricsRefreshing,
  polling: metricsPolling,
  lastUpdated: metricsLastUpdated,
  paused: metricsPaused,
  setIntervalSec: setMetricsInterval,
  refreshNow: refreshMetricsNow,
} = useAutoRefresh({
  refresh: poll,
  isActive: () => runIsActive.value,
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
});

watch(key, () => {
  cachedPreviewRunDir = null;
  jobArtifacts.value = [];
  job.value = null;
  fsRun.value = null;
  previewImages.value = [];
  metrics.value = {};
  void refreshMetricsNow();
});
</script>

<style scoped>
.mt-12 {
  margin-top: 12px;
}
.page-title-inline {
  font-weight: 600;
}
.continue-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
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

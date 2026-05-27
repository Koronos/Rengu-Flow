<template>
  <div>
    <el-page-header @back="goBack">
      <template #content>
        <span class="page-title-inline">{{ title }}</span>
      </template>
    </el-page-header>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mt-12" />

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

      <el-divider content-position="left">Signals</el-divider>
      <div class="signal-grid">
        <el-button
          v-for="s in SIGNALS"
          :key="s[0]"
          size="small"
          @click="sendSignal(s[0])"
        >
          {{ s[1] }}
        </el-button>
      </div>
    </el-card>

    <el-card shadow="never" class="mt-12">
      <template #header>Loss</template>
      <LossChart :scalars="metrics" />
    </el-card>

    <el-card v-if="mode === 'job'" shadow="never" class="mt-12">
      <template #header>Log</template>
      <pre class="log-pre">{{ log || "(waiting for output…)" }}</pre>
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

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useTensorboard } from "../composables/useTensorboard";
import LossChart from "../components/LossChart.vue";

const props = defineProps({
  mode: { type: String, required: true },
  name: { type: String, default: "" },
});

const route = useRoute();
const router = useRouter();
const { isMobile } = useBreakpoint();

const SIGNALS = [
  ["save", "Checkpoint"],
  ["save_quit", "Checkpoint + quit"],
  ["export_model", "Export model"],
  ["export_model_quit", "Export + quit"],
  ["preview", "Preview"],
];

const job = ref(null);
const fsRun = ref(null);
const jobArtifacts = ref([]);
const log = ref("");
const metrics = ref({});
const error = ref("");
const outputDir = ref("output");
const { tbLoading, tbStatus, refreshTbStatus, openTensorboard, stopTensorboard } = useTensorboard(
  () => job.value?.output_dir || outputDir.value || "output"
);
let pollTimer = null;
let logTimer = null;
let logOffset = 0;
let cachedPreviewRunDir = null;

const key = computed(() => (props.mode === "job" ? route.params.id : route.params.name));

const title = computed(() =>
  props.mode === "job" ? `Job ${key.value}` : `Run ${key.value}`
);

const runDir = computed(() => job.value?.run_dir || fsRun.value?.path);
const status = computed(() => fsRun.value?.status || null);
const artifacts = computed(() => fsRun.value?.artifacts?.length ? fsRun.value.artifacts : jobArtifacts.value);

function formatLoss(v) {
  return typeof v === "number" ? v.toFixed(6) : v;
}

function goBack() {
  router.push(props.mode === "job" ? "/jobs" : "/runs");
}

function goContinueTraining() {
  if (!runDir.value) return;
  router.push({ name: "configs", query: { continue_run: runDir.value } });
}

function openTensorboardForRun() {
  openTensorboard({ onError: (msg) => { error.value = msg; } }).catch(() => {});
}

function stopTensorboardForRun() {
  stopTensorboard({ onError: (msg) => { error.value = msg; } }).catch(() => {});
}

async function poll() {
  try {
    if (props.mode === "job") {
      job.value = await api.getJob(key.value);
      const m = await api.jobMetrics(key.value);
      metrics.value = m.scalars || {};
      fsRun.value = null;
      jobArtifacts.value = [];
      if (job.value?.run_dir) {
        if (job.value.run_dir !== cachedPreviewRunDir) {
          cachedPreviewRunDir = job.value.run_dir;
          jobArtifacts.value = [];
          try {
            const preview = await api.previewJobImport(job.value.run_dir);
            fsRun.value = preview.run || null;
          } catch {
            /* optional metadata */
          }
          try {
            const art = await api.jobArtifacts(key.value);
            jobArtifacts.value = art.artifacts || [];
          } catch {
            /* ignore */
          }
        }
      }
    } else {
      fsRun.value = await api.getFsRun(key.value, outputDir.value);
      const m = await api.fsMetrics(key.value, outputDir.value);
      metrics.value = m.scalars || {};
    }
  } catch (e) {
    error.value = String(e);
  }
  refreshTbStatus();
}

async function loadLog() {
  if (props.mode !== "job") return;
  try {
    const res = await fetch(`/api/v1/jobs/${key.value}/logs?offset=${logOffset}`);
    const data = await res.json();
    if (data.chunk) {
      log.value += data.chunk;
      logOffset = data.offset;
    }
  } catch {
    /* ignore */
  }
}

async function sendSignal(type) {
  error.value = "";
  try {
    if (props.mode === "job") {
      await api.sendJobSignal(key.value, type);
    } else {
      await api.fsSignal(key.value, type, outputDir.value);
    }
    ElMessage.success(`Signal "${type}" sent`);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

function startTimers() {
  log.value = "";
  logOffset = 0;
  cachedPreviewRunDir = null;
  jobArtifacts.value = [];
  refreshTbStatus();
  poll();
  pollTimer = setInterval(poll, 4000);
  if (props.mode === "job") {
    loadLog();
    logTimer = setInterval(loadLog, 1500);
  }
}

function stopTimers() {
  if (pollTimer) clearInterval(pollTimer);
  if (logTimer) clearInterval(logTimer);
  pollTimer = null;
  logTimer = null;
}

onMounted(startTimers);
onUnmounted(stopTimers);
watch(key, () => {
  stopTimers();
  job.value = null;
  fsRun.value = null;
  startTimers();
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
.signal-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.mono {
  font-family: ui-monospace, monospace;
  font-size: 12px;
}
</style>

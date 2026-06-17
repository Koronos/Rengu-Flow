<template>
  <div class="prep-jobs-page page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Prepare datasets: tagging, captioning, watermark cleanup, tag editing</p>
        <el-text v-if="stats.running || stats.pending" type="info" class="page-head-meta stats-line">
          <span v-if="stats.running" class="stat-running">{{ stats.running }} running</span>
          <span v-if="stats.pending">{{ stats.pending }} in queue</span>
        </el-text>
      </div>
      <el-space wrap>
        <el-button type="primary" :icon="MagicStick" @click="goNewJob('tag')">New tag job</el-button>
        <el-button :icon="ChatLineRound" @click="goNewJob('caption')">New caption job</el-button>
        <el-button :icon="Delete" @click="goNewJob('clean')">New clean job</el-button>
        <el-button :icon="Edit" @click="$router.push('/prep/tags')">Tag editor</el-button>
      </el-space>
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mt-12" />

    <!-- Live panel for running job -->
    <PrepJobLivePanel
      v-if="activeJob"
      :job-id="String(activeJob.id)"
      :stage="jobStage(activeJob)"
      class="page-section"
      @stopped="refresh"
    />

    <el-card shadow="never" class="page-section">
      <template #header>
        <div class="card-header-row">
          <span>Dataset Studio</span>
          <el-button size="small" :icon="Refresh" circle :loading="loading" @click="refresh" />
        </div>
      </template>

      <el-empty
        v-if="!loading && !jobs.length"
        description="No jobs yet. Use the buttons above to launch one."
        :image-size="56"
      />

      <div v-else class="job-rows">
        <div
          v-for="job in jobs"
          :key="job.id"
          class="job-row"
          :class="{ 'job-row--active': isActive(job), 'job-row--expanded': expandedId === job.id }"
        >
          <div class="job-row__main" @click="toggleExpand(job)">
            <el-tag :type="stateTag(job.state)" size="small" effect="dark">{{ job.state }}</el-tag>
            <el-tag size="small" type="info" effect="plain">{{ jobStage(job) }}</el-tag>
            <span class="job-row__label" :title="jobLabelFull(job)">{{ jobLabel(job) }}</span>
            <span class="job-row__time">{{ formatTime(job.started_at || job.finished_at) }}</span>
          </div>
          <el-space class="job-row__actions" @click.stop>
            <el-tooltip v-if="isActive(job)" content="Stop" :show-after="300">
              <el-button
                size="small"
                circle
                type="danger"
                plain
                :icon="VideoPause"
                @click.stop="stopJob(job)"
              />
            </el-tooltip>
            <el-tooltip v-if="isTerminal(job)" content="Re-queue job" :show-after="300">
              <el-button
                size="small"
                :icon="RefreshRight"
                @click.stop="requeueJob(job)"
              >Re-queue</el-button>
            </el-tooltip>
            <el-tooltip content="Create a training dataset from this folder" :show-after="300">
              <el-button
                size="small"
                :icon="Files"
                @click.stop="openDatasetFromJob(job)"
              >Generate dataset</el-button>
            </el-tooltip>
            <el-tooltip v-if="isTerminal(job)" content="Delete job" :show-after="300">
              <el-button
                size="small"
                circle
                :icon="Delete"
                @click.stop="deleteJob(job)"
              />
            </el-tooltip>
          </el-space>

          <!-- Expanded panel: live progress + report -->
          <div v-if="expandedId === job.id" class="job-row__expanded-panel">
            <PrepJobLivePanel
              v-if="isActive(job)"
              :job-id="String(job.id)"
              :stage="jobStage(job)"
              class="job-expand-live"
              @stopped="refresh"
            />

            <div v-if="reportData(job.id)" class="job-report">
              <el-divider content-position="left">Report</el-divider>
              <el-descriptions :column="isMobile ? 1 : 3" border size="small">
                <el-descriptions-item
                  v-for="(val, key) in reportCountFields(job.id)"
                  :key="key"
                  :label="String(key)"
                >
                  {{ val }}
                </el-descriptions-item>
              </el-descriptions>
              <el-alert
                v-if="reportFailed(job.id).length"
                type="error"
                show-icon
                :closable="false"
                class="mt-8"
                title="Failed items"
              >
                <ul class="report-failed-list">
                  <li v-for="f in reportFailed(job.id)" :key="f">{{ f }}</li>
                </ul>
              </el-alert>
            </div>
            <el-text v-else-if="!isActive(job)" size="small" type="info" class="mt-8">
              No report available.
            </el-text>

            <div v-if="!isActive(job)" class="job-log">
              <el-divider content-position="left">
                Log
                <el-button
                  size="small"
                  text
                  :icon="Refresh"
                  :loading="logLoading[job.id]"
                  @click="loadJobLog(job.id, true)"
                />
              </el-divider>
              <pre v-if="logs[job.id]" class="job-log__pre">{{ logs[job.id] }}</pre>
              <el-text v-else size="small" type="info">
                {{ logLoading[job.id] ? "Loading log…" : "No log captured for this job." }}
              </el-text>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  ChatLineRound,
  Delete,
  Edit,
  Files,
  MagicStick,
  Refresh,
  RefreshRight,
  VideoPause,
} from "@element-plus/icons-vue";
import { api } from "../api";
import PrepJobLivePanel from "../components/PrepJobLivePanel.vue";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useDatasetFormModal } from "../composables/useDatasetFormModal";
import { useJobsEvents } from "../composables/useJobsEvents";
import { formatError } from "../lib/formatError";
import { DEFAULT_DATASET_TOML } from "../stores/datasetEditor";
import type { JobRecord, PrepStage } from "../types/api";

const router = useRouter();
const { isMobile } = useBreakpoint();
const modal = useDatasetFormModal();

const jobs = ref<JobRecord[]>([]);
const stats = ref({ running: 0, pending: 0 });
const loading = ref(false);
const error = ref("");
const expandedId = ref<string | null>(null);
const reports = ref<Record<string, Record<string, unknown> | null>>({});
const logs = ref<Record<string, string>>({});
const logLoading = ref<Record<string, boolean>>({});

async function loadJobLog(jobId: string | number, force = false): Promise<void> {
  const id = String(jobId);
  if (!force && logs.value[id]) return;
  logLoading.value = { ...logLoading.value, [id]: true };
  try {
    const { chunk } = await api.jobLogs(id, 0);
    logs.value = { ...logs.value, [id]: chunk };
  } catch {
    // leave empty -> "No log captured" text
  } finally {
    logLoading.value = { ...logLoading.value, [id]: false };
  }
}


const activeJob = ref<JobRecord | null>(null);

function jobStage(job: JobRecord): PrepStage {
  // Stage is stored in extra_args[0] or in kind
  if (job.extra_args?.[0]) return job.extra_args[0] as PrepStage;
  if (job.kind) return job.kind as PrepStage;
  return "tag";
}

/** Extract dataset path from TOML config content (^path = "..." multiline). */
function datasetPathFromConfig(content: string | undefined): string {
  if (!content) return "";
  const m = content.match(/^path\s*=\s*"(.*)"/m);
  return m ? m[1] : "";
}

function pathBasename(p: string): string {
  if (!p) return "";
  return p.replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() || p;
}

function jobLabel(job: JobRecord): string {
  const stage = jobStage(job);
  const datasetPath = datasetPathFromConfig(job.config_content);
  const base = datasetPath ? pathBasename(datasetPath) : (job.run_dir ? pathBasename(job.run_dir) : "");
  if (base) return `${base} · ${stage}`;
  return `${stage} job #${job.id}`;
}

function jobLabelFull(job: JobRecord): string {
  const datasetPath = datasetPathFromConfig(job.config_content) || job.run_dir || job.config_path || "";
  return datasetPath || `job #${job.id}`;
}

function isActive(job: JobRecord): boolean {
  return job.state === "running" || job.state === "stopping" || job.state === "pending";
}

function isTerminal(job: JobRecord): boolean {
  return job.state === "finished" || job.state === "failed" || job.state === "stopped";
}

function stateTag(state: string): "primary" | "success" | "warning" | "info" | "danger" {
  if (state === "running" || state === "stopping") return "success";
  if (state === "pending") return "warning";
  if (state === "finished") return "info";
  if (state === "stopped") return "warning";
  if (state === "failed") return "danger";
  return "info";
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "";
  return String(iso).slice(0, 19).replace("T", " ");
}

function reportData(id: string): Record<string, unknown> | null {
  return reports.value[id] ?? null;
}

function reportCountFields(id: string): Record<string, unknown> {
  const r = reportData(id);
  if (!r) return {};
  const excluded = new Set(["failed", "errors"]);
  return Object.fromEntries(
    Object.entries(r).filter(([k]) => !excluded.has(k))
  );
}

function reportFailed(id: string): string[] {
  const r = reportData(id);
  if (!r) return [];
  const f = r.failed;
  if (Array.isArray(f)) return f.map(String);
  return [];
}

async function fetchReport(job: JobRecord): Promise<void> {
  if (reports.value[job.id] !== undefined) return;
  try {
    const res = await api.prepJobReport(job.id);
    reports.value = { ...reports.value, [job.id]: res.report };
  } catch {
    reports.value = { ...reports.value, [job.id]: null };
  }
}

function toggleExpand(job: JobRecord): void {
  if (expandedId.value === job.id) {
    expandedId.value = null;
    return;
  }
  expandedId.value = job.id;
  if (isTerminal(job)) {
    void fetchReport(job);
    void loadJobLog(job.id);
  }
}

async function refresh(): Promise<void> {
  loading.value = true;
  error.value = "";
  try {
    const data = await api.prepJobs();
    jobs.value = data.jobs || [];
    stats.value = data.stats || { running: 0, pending: 0 };
    activeJob.value = jobs.value.find(isActive) ?? null;
    // refresh report for expanded terminal job
    if (expandedId.value) {
      const j = jobs.value.find((x) => x.id === expandedId.value);
      if (j && isTerminal(j)) void fetchReport(j);
    }
  } catch (e) {
    error.value = formatError(e);
  } finally {
    loading.value = false;
  }
}

async function stopJob(job: JobRecord): Promise<void> {
  try {
    await api.stopJob(job.id);
    ElMessage.info("Stop requested");
    await refresh();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function deleteJob(job: JobRecord): Promise<void> {
  try {
    await ElMessageBox.confirm(
      "Delete this job from the list?",
      "Delete job",
      { type: "warning", confirmButtonText: "Delete", cancelButtonText: "Cancel" }
    );
  } catch {
    return;
  }
  try {
    await api.deleteJob(job.id);
    ElMessage.success("Deleted");
    if (expandedId.value === job.id) expandedId.value = null;
    await refresh();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function requeueJob(job: JobRecord): Promise<void> {
  try {
    await api.requeuePrepJob(job.id, { start_now: false });
    ElMessage.success(
      "Job re-queued — it resumes where it stopped (already-processed images are skipped unless Overwrite was on)"
    );
    await refresh();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

function buildDatasetToml(folderPath: string): string {
  return DEFAULT_DATASET_TOML + `\n[[directory]]\npath = "${folderPath}"\nnum_repeats = 1\n`;
}

async function openDatasetFromJob(job: JobRecord): Promise<void> {
  // The report wins: a clean job's dataset is its OUTPUT folder (cleaned copies),
  // which only the report knows. Fall back to the job config's source path.
  let report = reports.value[job.id];
  if (!report && isTerminal(job)) {
    try {
      const res = await api.prepJobReport(job.id);
      reports.value = { ...reports.value, [job.id]: res.report };
      report = res.report;
    } catch {
      // ignore — config fallback below
    }
  }
  let folder = report ? String(report.output_dir ?? report.path ?? "") : "";
  if (!folder) {
    folder = datasetPathFromConfig(job.config_content);
  }
  modal.openCreate({ initialToml: buildDatasetToml(folder || "") });
}

function goNewJob(stage: PrepStage): void {
  router.push({ name: "prep-new", params: { stage } });
}

onMounted(() => void refresh());

// Refresh the list on server-pushed job changes (create/update/delete, including a run the queue
// poller transitions to finished/failed) instead of polling GET /jobs?kind=prep on a timer.
useJobsEvents(() => void refresh());
</script>

<style scoped>
.page-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.stats-line {
  display: flex;
  gap: var(--rf-space-sm);
}
.stat-running {
  color: var(--el-color-success);
  font-weight: 600;
}
.page-section {
  margin-top: var(--rf-space-md);
}
.card-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.job-rows {
  display: flex;
  flex-direction: column;
  gap: 0;
}
.job-row {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  background: var(--el-bg-color);
  margin-bottom: 6px;
  overflow: hidden;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.job-row--active {
  border-color: var(--el-color-success-light-5);
  background: var(--el-color-success-light-9);
}
.job-row__main {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  flex: 1;
  min-width: 0;
}
.job-row__main:hover {
  background: var(--el-fill-color-light);
}
.job-row__label {
  font-weight: 500;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.job-row__time {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  white-space: nowrap;
}
.job-row__actions {
  flex-shrink: 0;
  padding-right: 8px;
}
.job-row__expanded-panel {
  flex-basis: 100%;
  padding: 12px 16px 16px;
  border-top: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-blank);
}
.job-expand-live {
  margin-bottom: 12px;
}
.job-report {
  margin-top: 4px;
}
.report-failed-list {
  margin: 4px 0 0 0;
  padding-left: 20px;
  font-size: 12px;
  font-family: var(--rf-font-mono);
}
.mt-8 {
  margin-top: 8px;
}
.mt-12 {
  margin-bottom: 12px;
}
.job-log__pre {
  max-height: 320px;
  overflow: auto;
  background: var(--el-fill-color-darker);
  border-radius: var(--el-border-radius-base);
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>

<template>
  <div class="jobs-page">
    <div class="page-head">
      <h2 class="page-title">Training jobs</h2>
      <el-text type="info" class="stats-line">
        <span v-if="stats.running" class="stat-running">{{ stats.running }} running</span>
        <span v-if="stats.pending">{{ stats.pending }} in queue</span>
      </el-text>
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <el-alert type="info" :closable="false" show-icon class="mb-12 workflow-alert">
      <template #title>Recommended workflow</template>
      <ol class="workflow-list">
        <li>
          <router-link to="/datasets">Datasets</router-link> — image folders (optional if you use an example path)
        </li>
        <li>
          <a href="#" class="workflow-link" @click.prevent="goPickConfig">Configs</a>
          — training TOML (required before a job)
        </li>
        <li><strong>Jobs</strong> — queue or start training here</li>
      </ol>
    </el-alert>

    <el-card shadow="hover" class="mb-12 launch-card">
      <template #header>
        <span>Add training job</span>
      </template>
      <el-form label-position="top" class="launch-form">
        <el-form-item label="Training config" required>
          <div v-if="!hasAnyConfig" class="config-block">
            <el-empty description="No training configs yet" :image-size="56">
              <el-button type="primary" @click="goCreateConfig">Create your first config</el-button>
            </el-empty>
          </div>
          <div v-else-if="!configId" class="config-block config-block--empty">
            <p class="config-hint">
              Pick a config from the library so you can review and edit it before training.
            </p>
            <el-button type="primary" @click="goPickConfig">Choose config in library</el-button>
          </div>
          <div v-else class="config-block config-block--selected">
            <div class="config-selected-row">
              <code class="config-id">{{ configId }}</code>
              <el-tag v-if="configValid === true" type="success" size="small">Valid</el-tag>
              <el-tag v-else-if="configValid === false" type="danger" size="small">Needs fixes</el-tag>
            </div>
            <el-space wrap>
              <el-button @click="goPickConfig">Change</el-button>
              <el-button type="primary" plain @click="goEditConfig">Edit before run</el-button>
              <el-button :icon="CircleCheck" :loading="validating" @click="checkConfig">
                Validate
              </el-button>
            </el-space>
          </div>
        </el-form-item>

        <el-row :gutter="16">
          <el-col :xs="12" :sm="6" :md="4">
            <el-form-item label="GPUs">
              <el-input-number v-model="numGpus" :min="1" :max="64" class="w-full" />
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="24" :md="10">
            <el-form-item label="Resume folder">
              <el-input
                v-model="resumeFrom"
                placeholder="output/20250217_14-30-00"
                clearable
                class="w-full"
              />
            </el-form-item>
          </el-col>
        </el-row>
        <div class="launch-actions">
          <el-button type="primary" :icon="Plus" :disabled="!canLaunch" @click="enqueue">
            Add to queue
          </el-button>
          <el-button :icon="VideoPlay" :disabled="!canLaunch" @click="startNow">
            Start now
          </el-button>
        </div>
        <el-text type="info" size="small" class="hint">
          One job runs at a time. Queued jobs start automatically when the current run finishes.
        </el-text>
      </el-form>
    </el-card>

    <el-card shadow="hover" class="mb-12 import-card">
      <template #header>
        <span>Import script run</span>
      </template>
      <p class="import-hint">
        Already trained with <code>renga-flow</code> from the terminal? Register an output folder so it
        appears in job history with metrics, signals, and optional config/dataset library entries.
      </p>
      <el-button type="primary" plain :icon="FolderOpened" @click="openImportDialog">
        Import run folder…
      </el-button>
    </el-card>

    <section v-if="runningJobs.length" class="job-section">
      <h3 class="section-title">
        <span class="pulse-dot" />
        Running
      </h3>
      <div class="job-cards">
        <el-card
          v-for="job in runningJobs"
          :key="job.id"
          shadow="always"
          class="job-card job-card--active"
          @click="goJob(job.id)"
        >
          <div class="job-card-top">
            <el-tag type="success" effect="dark" size="small">{{ job.state }}</el-tag>
            <code class="job-id">{{ job.id }}</code>
          </div>
          <div class="job-meta">{{ job.config_id || "—" }} · {{ job.num_gpus }} GPU</div>
          <div class="job-meta muted">PID {{ job.pid || "—" }}</div>
          <el-button
            type="danger"
            size="small"
            :icon="VideoPause"
            class="job-stop"
            @click.stop="stop(job.id)"
          >
            Stop
          </el-button>
        </el-card>
      </div>
    </section>

    <section v-if="queueJobs.length" class="job-section">
      <h3 class="section-title">Queue</h3>
      <el-table :data="queueJobs" stripe size="small" class="queue-table">
        <el-table-column label="#" width="48">
          <template #default="{ $index }">{{ $index + 1 }}</template>
        </el-table-column>
        <el-table-column prop="config_id" label="Config" min-width="120" show-overflow-tooltip />
        <el-table-column prop="num_gpus" label="GPUs" width="70" />
        <el-table-column prop="resume_from" label="Resume" min-width="120" show-overflow-tooltip />
        <el-table-column label="Actions" width="280" fixed="right">
          <template #default="{ row }">
            <el-button-group size="small">
              <el-button :icon="Top" @click="move(row.id, 'up')" />
              <el-button :icon="Bottom" @click="move(row.id, 'down')" />
              <el-button :icon="Edit" @click="openEdit(row)" />
              <el-button type="primary" :icon="VideoPlay" @click="startQueuedNow(row.id)">
                Run
              </el-button>
              <el-button type="danger" :icon="Delete" @click="removeQueued(row.id)" />
            </el-button-group>
          </template>
        </el-table-column>
      </el-table>
    </section>

    <el-collapse v-if="historyJobs.length" class="history-collapse">
      <el-collapse-item title="History" name="history">
        <el-table :data="historyJobs" stripe size="small" @row-click="(r) => goJob(r.id)">
          <el-table-column prop="id" label="ID" min-width="100" />
          <el-table-column prop="state" label="State" width="90">
            <template #default="{ row }">
              <el-tag :type="stateTag(row.state)" size="small">{{ row.state }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="config_id" label="Config" min-width="100" />
          <el-table-column label="Run folder" min-width="120" show-overflow-tooltip>
            <template #default="{ row }">
              {{ runFolderLabel(row) }}
            </template>
          </el-table-column>
          <el-table-column label="Finished" min-width="140">
            <template #default="{ row }">
              {{ formatTime(row.finished_at || row.started_at) }}
            </template>
          </el-table-column>
        </el-table>
      </el-collapse-item>
    </el-collapse>

    <el-empty
      v-if="!runningJobs.length && !queueJobs.length && !historyJobs.length"
      description="No jobs yet — add one to the queue above."
    />

    <el-dialog
      v-model="importOpen"
      title="Import training run folder"
      width="640px"
      destroy-on-close
      @closed="resetImportDialog"
    >
      <el-form label-position="top">
        <el-form-item label="Output directory (browse runs)">
          <el-input v-model="importOutputDir" placeholder="output" class="w-full" @change="loadImportCandidates">
            <template #append>
              <el-button @click="loadImportCandidates">Scan</el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item v-if="importCandidates.length" label="Runs under output dir">
          <el-select
            v-model="importRunPath"
            filterable
            class="w-full"
            placeholder="Pick a run folder"
            @change="onImportPathPicked"
          >
            <el-option
              v-for="r in importCandidates"
              :key="r.path"
              :label="r.name + (r.already_imported ? ' (imported)' : '')"
              :value="r.path"
              :disabled="r.already_imported"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Run folder path" required>
          <el-input
            v-model="importRunPath"
            placeholder="output/20250217_14-30-00 or absolute path"
            class="w-full"
          />
        </el-form-item>
        <el-button :loading="importPreviewLoading" @click="previewImportRun">Preview</el-button>
      </el-form>

      <el-alert
        v-if="importPreview?.already_imported"
        type="warning"
        :closable="false"
        show-icon
        class="mt-12"
        title="This folder is already linked to a job"
      />

      <el-descriptions v-if="importPreview?.run" :column="1" border size="small" class="mt-12">
        <el-descriptions-item label="Folder">{{ importPreview.run.name }}</el-descriptions-item>
        <el-descriptions-item label="Config">
          {{ importPreview.config_path || "—" }}
        </el-descriptions-item>
        <el-descriptions-item label="Artifacts">
          {{ importPreview.run.artifacts?.length || 0 }}
        </el-descriptions-item>
        <el-descriptions-item label="TensorBoard">
          {{ importPreview.run.has_tensorboard ? "yes" : "no" }}
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />
      <el-checkbox v-model="importForm.import_config">Add training config to library</el-checkbox>
      <el-form v-if="importForm.import_config" label-position="top" class="mt-8">
        <el-form-item label="Library config id">
          <el-input v-model="importForm.config_id" placeholder="auto from folder name" />
        </el-form-item>
      </el-form>
      <el-checkbox v-model="importForm.import_dataset">Add dataset TOML to library (if present)</el-checkbox>
      <el-form v-if="importForm.import_dataset" label-position="top" class="mt-8">
        <el-form-item label="Library dataset id">
          <el-input v-model="importForm.dataset_id" placeholder="auto" />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="importOpen = false">Cancel</el-button>
        <el-button
          type="primary"
          :loading="importSaving"
          :disabled="!importRunPath.trim()"
          @click="confirmImportRun"
        >
          Import
        </el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="editOpen" title="Edit queued job" width="480px" destroy-on-close>
      <el-form label-position="top">
        <el-form-item label="Config">
          <el-select
            v-model="editForm.config_id"
            filterable
            remote
            reserve-keyword
            placeholder="Search configs…"
            :remote-method="searchConfigOptions"
            class="w-full"
          >
            <el-option
              v-for="c in configOptions"
              :key="c.id"
              :label="c.model_type ? `${c.id} (${c.model_type})` : c.id"
              :value="c.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="GPUs">
          <el-input-number v-model="editForm.num_gpus" :min="1" :max="64" class="w-full" />
        </el-form-item>
        <el-form-item label="Resume folder">
          <el-input v-model="editForm.resume_from" clearable placeholder="optional" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editOpen = false">Cancel</el-button>
        <el-button type="primary" @click="saveEdit">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  Bottom,
  CircleCheck,
  Delete,
  Edit,
  FolderOpened,
  Plus,
  Top,
  VideoPause,
  VideoPlay,
} from "@element-plus/icons-vue";
import { api } from "../api";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";

const router = useRouter();

const jobs = ref([]);
const stats = ref({ running: 0, pending: 0 });
const hasAnyConfig = ref(false);
const configOptions = ref([]);
const configId = ref(getJobConfigId());
const configValid = ref(null);
const validating = ref(false);
const numGpus = ref(1);
const resumeFrom = ref("");
const error = ref("");
let timer = null;

const importOpen = ref(false);
const importRunPath = ref("");
const importOutputDir = ref("output");
const importCandidates = ref([]);
const importPreview = ref(null);
const importPreviewLoading = ref(false);
const importSaving = ref(false);
const importForm = reactive({
  import_config: true,
  import_dataset: true,
  config_id: "",
  dataset_id: "",
});

const editOpen = ref(false);
const editJobId = ref("");
const editForm = reactive({
  config_id: "",
  num_gpus: 1,
  resume_from: "",
});

const runningJobs = computed(() =>
  jobs.value.filter((j) => j.state === "running" || j.state === "stopping")
);

const queueJobs = computed(() =>
  jobs.value
    .filter((j) => j.state === "pending")
    .sort((a, b) => (a.queue_position ?? 999) - (b.queue_position ?? 999))
);

const historyJobs = computed(() =>
  jobs.value.filter((j) => !["running", "stopping", "pending"].includes(j.state))
);

const canLaunch = computed(() => Boolean(configId.value && hasAnyConfig.value));

function stateTag(state) {
  if (state === "finished") return "info";
  if (state === "stopped") return "warning";
  if (state === "failed") return "danger";
  return "";
}

function formatTime(iso) {
  return (iso || "").slice(0, 19).replace("T", " ");
}

function runFolderLabel(row) {
  if (!row.run_dir) return "—";
  const parts = String(row.run_dir).replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || row.run_dir;
}

function openImportDialog() {
  importOpen.value = true;
  loadImportCandidates().catch(() => {});
}

function resetImportDialog() {
  importRunPath.value = "";
  importPreview.value = null;
  importForm.import_config = true;
  importForm.import_dataset = true;
  importForm.config_id = "";
  importForm.dataset_id = "";
}

async function loadImportCandidates() {
  const data = await api.listImportCandidates(importOutputDir.value || "output");
  importCandidates.value = data.runs || [];
}

function onImportPathPicked(path) {
  importRunPath.value = path;
  previewImportRun();
}

async function previewImportRun() {
  const path = importRunPath.value.trim();
  if (!path) return;
  importPreviewLoading.value = true;
  importPreview.value = null;
  try {
    const data = await api.previewJobImport(path);
    importPreview.value = data;
    if (!importForm.config_id) {
      importForm.config_id = data.suggested_config_id || "";
    }
    if (!importForm.dataset_id) {
      importForm.dataset_id = data.suggested_dataset_id || "";
    }
  } catch (e) {
    ElMessage.error(String(e));
  } finally {
    importPreviewLoading.value = false;
  }
}

async function confirmImportRun() {
  const path = importRunPath.value.trim();
  if (!path) return;
  importSaving.value = true;
  try {
    const job = await api.importJobFromRun({
      run_path: path,
      import_config: importForm.import_config,
      config_id: importForm.config_id.trim() || undefined,
      import_dataset: importForm.import_dataset,
      dataset_id: importForm.dataset_id.trim() || undefined,
    });
    ElMessage.success("Run imported");
    importOpen.value = false;
    await refresh();
    router.push(`/jobs/${job.id}`);
  } catch (e) {
    ElMessage.error(String(e));
  } finally {
    importSaving.value = false;
  }
}

async function refreshConfigAvailability() {
  try {
    const r = await api.searchConfigs({ q: "", page: 1, page_size: 1 });
    hasAnyConfig.value = (r.total ?? 0) > 0;
  } catch {
    hasAnyConfig.value = false;
  }
}

async function searchConfigOptions(query) {
  try {
    const r = await api.searchConfigs({ q: query || "", page: 1, page_size: 30 });
    configOptions.value = r.items || [];
  } catch {
    configOptions.value = [];
  }
}

function syncConfigSelection() {
  const stored = getJobConfigId();
  if (stored) {
    configId.value = stored;
  }
}

async function refresh() {
  const j = await api.listJobs();
  jobs.value = j.jobs || [];
  stats.value = j.stats || { running: 0, pending: 0 };
  await refreshConfigAvailability();
  syncConfigSelection();
  if (configId.value) checkConfig();
}

function goPickConfig() {
  router.push({ name: "configs", query: { pick: "job" } });
}

function goEditConfig() {
  if (!configId.value) return;
  router.push({ name: "configs", query: { config: configId.value } });
}

function goCreateConfig() {
  router.push({ name: "configs", query: { new: "1" } });
}

async function checkConfig() {
  if (!configId.value) return;
  validating.value = true;
  configValid.value = null;
  try {
    const { content } = await api.getConfig(configId.value);
    const r = await api.validate(content);
    configValid.value = r.ok;
    if (!r.ok) {
      ElMessage.warning(r.error || "Config is not valid yet");
    }
  } catch (e) {
    configValid.value = false;
    ElMessage.error(String(e));
  } finally {
    validating.value = false;
  }
}

onMounted(() => {
  refresh().catch((e) => { error.value = String(e); });
  timer = setInterval(() => refresh().catch(() => {}), 3000);
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
});

function goJob(id) {
  router.push(`/jobs/${id}`);
}

async function enqueue() {
  error.value = "";
  if (!canLaunch.value) return;
  setJobConfigId(configId.value);
  try {
    await api.startJob({
      config_id: configId.value,
      num_gpus: numGpus.value,
      resume_from: resumeFrom.value || undefined,
      enqueue: true,
      start_immediately: false,
    });
    ElMessage.success("Added to queue");
    await refresh();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function startNow() {
  error.value = "";
  if (!canLaunch.value) return;
  setJobConfigId(configId.value);
  try {
    await api.startJob({
      config_id: configId.value,
      num_gpus: numGpus.value,
      resume_from: resumeFrom.value || undefined,
      enqueue: false,
      start_immediately: true,
    });
    ElMessage.success("Job started or queued at front");
    await refresh();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function stop(id) {
  await api.stopJob(id);
  ElMessage.info("Stop requested");
  await refresh();
}

async function move(id, direction) {
  await api.moveJobQueue(id, direction);
  await refresh();
}

async function startQueuedNow(id) {
  await api.startJobNow(id);
  ElMessage.success("Moved to front");
  await refresh();
}

async function removeQueued(id) {
  await ElMessageBox.confirm("Remove this job from the queue?", "Confirm", { type: "warning" });
  await api.deleteJob(id);
  ElMessage.success("Removed");
  await refresh();
}

function openEdit(row) {
  editJobId.value = row.id;
  editForm.config_id = row.config_id || "";
  editForm.num_gpus = row.num_gpus || 1;
  editForm.resume_from = row.resume_from || "";
  searchConfigOptions(row.config_id || "");
  editOpen.value = true;
}

async function saveEdit() {
  await api.updateJob(editJobId.value, {
    config_id: editForm.config_id,
    num_gpus: editForm.num_gpus,
    resume_from: editForm.resume_from || null,
  });
  editOpen.value = false;
  ElMessage.success("Queue job updated");
  await refresh();
}
</script>

<style scoped>
.jobs-page {
  max-width: 100%;
}
.page-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 16px;
}
.page-title {
  margin: 0;
}
.stats-line {
  display: flex;
  gap: 12px;
}
.stat-running {
  color: var(--el-color-success);
  font-weight: 600;
}
.mb-12 {
  margin-bottom: 12px;
}
.launch-card :deep(.el-card__header) {
  font-weight: 600;
}
.launch-form :deep(.el-form-item) {
  margin-bottom: 0;
}
.launch-form :deep(.el-form-item__label) {
  padding-bottom: 4px;
  line-height: 1.3;
}
.launch-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}
.hint {
  display: block;
}
.job-section {
  margin-bottom: 20px;
}
.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1rem;
  margin: 0 0 10px;
  font-weight: 600;
}
.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--el-color-success);
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}
.job-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
}
.job-card {
  cursor: pointer;
  position: relative;
}
.job-card--active {
  border-color: var(--el-color-success);
}
.job-card-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.job-id {
  font-size: 12px;
}
.job-meta {
  font-size: 13px;
  margin-bottom: 4px;
}
.job-meta.muted {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.job-stop {
  margin-top: 8px;
}
.queue-table {
  width: 100%;
}
.history-collapse {
  margin-top: 8px;
}
.w-full {
  width: 100%;
}
.workflow-alert :deep(.el-alert__content) {
  width: 100%;
}
.workflow-list {
  margin: 8px 0 0;
  padding-left: 1.2rem;
  font-size: 13px;
  line-height: 1.6;
}
.workflow-list a,
.workflow-link {
  color: var(--el-color-primary);
  text-decoration: none;
}
.workflow-list a:hover,
.workflow-link:hover {
  text-decoration: underline;
}
.config-block {
  padding: 12px;
  border-radius: var(--el-border-radius-base);
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color-lighter);
}
.config-block--empty .config-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.config-selected-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.config-id {
  font-size: 14px;
  font-weight: 600;
}
.import-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.5;
}
.import-hint code {
  font-size: 12px;
}
.mt-8 {
  margin-top: 8px;
}
.mt-12 {
  margin-top: 12px;
}
</style>

<template>
  <div class="jobs-page page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Launch jobs and monitor runs</p>
        <el-text v-if="stats.running || stats.pending" type="info" class="page-head-meta stats-line">
          <span v-if="stats.running" class="stat-running">{{ stats.running }} running</span>
          <span v-if="stats.pending">{{ stats.pending }} in queue</span>
        </el-text>
      </div>
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />
    <el-alert
      v-if="pollWarning && !error"
      type="warning"
      :title="pollWarning"
      show-icon
      class="mb-12"
    />

    <TrainLivePanel
      class="page-section"
      :run="activeRun ?? undefined"
      :metrics-loading="liveMetricsLoading && liveUseHttpFallback"
      :log-text="liveLogText"
      :stream-status="liveStreamStatus"
      :stream-error="liveStreamError"
      @open-detail="openRun"
      @stop="stop"
      @signal="sendRunSignal"
    >
      <template v-if="hasLiveRun" #header-extra>
        <AutoRefreshBar
          :interval-sec="liveIntervalSec"
          :refreshing="liveRefreshing"
          :polling="livePolling"
          :last-updated="liveLastUpdated"
          :paused="livePaused"
          @update:interval-sec="setLiveInterval"
          @refresh="refreshLiveNow"
        />
      </template>
    </TrainLivePanel>

    <el-card shadow="never" class="page-section">
      <template #header>
        <div class="launch-head">
          <span>New training run</span>
          <el-button size="small" :icon="FolderOpened" @click="openImportDialog">Import existing run</el-button>
        </div>
      </template>
      <el-form label-position="top" class="launch-form">
        <el-form-item label="Training config" required>
          <div v-if="!hasAnyConfig" class="page-panel config-block--compact">
            <span class="page-hint">No training configs yet.</span>
            <el-button type="primary" size="small" @click="goCreateConfig">Create config</el-button>
          </div>
          <div v-else-if="!configId" class="page-panel">
            <p class="page-hint">
              Pick a config from the library so you can review and edit it before training.
            </p>
            <el-button type="primary" @click="goPickConfig">Choose config in library</el-button>
          </div>
          <div v-else class="page-panel">
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
              <el-input-number
                v-model="numGpus"
                :min="1"
                :max="64"
                class="w-full"
                aria-label="Number of GPUs for this job"
              />
              <el-text type="info" size="small" class="page-hint hint">
                DeepSpeed GPU count for this launch (default from local config is 1).
              </el-text>
            </el-form-item>
          </el-col>
          <el-col :xs="24" :sm="24" :md="10">
            <el-form-item label="Resume folder">
              <el-input
                v-model="resumeFrom"
                placeholder="20250217_14-30-00"
                clearable
                class="w-full"
                aria-label="Resume run folder name under output_dir"
              />
              <el-text type="info" size="small" class="page-hint hint">
                Folder name under your config’s <code>output_dir</code> (not a full path). Loads
                <code>latest</code> checkpoint via <code>--resume_from_checkpoint</code>.
              </el-text>
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item class="cache-options">
          <el-checkbox v-model="trustCache" :disabled="regenerateCache">
            Use existing cache (skip rebuild)
          </el-checkbox>
          <el-checkbox v-model="regenerateCache" class="cache-option-second">
            Regenerate cache
          </el-checkbox>
          <el-text type="info" size="small" class="page-hint hint block-hint">
            Default: build or refresh cache as needed. Enable “Use existing cache” after a
            successful <strong>Build cache</strong> run (<code>--trust_cache</code>). Use
            “Regenerate cache” after changing images or captions.
          </el-text>
        </el-form-item>

        <div class="launch-actions">
          <el-button type="primary" :icon="Plus" :disabled="!canLaunch" @click="enqueue">
            Add to queue
          </el-button>
          <el-button :icon="VideoPlay" :disabled="!canLaunch" @click="startNow">
            Start now
          </el-button>
        </div>
        <el-text type="info" size="small" class="page-hint hint">
          One job runs at a time. Queued jobs start automatically when the current run finishes.
        </el-text>

        <el-divider content-position="left">Dataset cache only</el-divider>
        <div class="launch-actions">
          <el-button :disabled="!canLaunch" @click="enqueueCache">
            Build cache / Generar caché (queue)
          </el-button>
          <el-button :icon="VideoPlay" :disabled="!canLaunch" @click="startCacheNow">
            Build cache (start now)
          </el-button>
        </div>
        <el-text type="info" size="small" class="page-hint hint">
          Pre-compute latents and text embeddings without training (<code>--cache_only</code>).
          Recommended before long Cosmos runs. Does not use the resume folder.
        </el-text>
      </el-form>
    </el-card>

    <div class="page-toolbar">
      <el-input
        v-model="listQuery"
        clearable
        placeholder="Search config, run folder…"
        class="page-toolbar-search"
        @clear="loadRuns(1)"
      />
      <el-select v-model="stateFilter" clearable placeholder="All states" class="page-toolbar-filter" @change="loadRuns(1)">
        <el-option label="Active" value="active" />
        <el-option label="Queued" value="queued" />
        <el-option label="Finished" value="finished" />
        <el-option label="On disk only" value="disk" />
      </el-select>
      <el-tooltip content="Refresh runs list" :show-after="300">
        <el-button
          size="small"
          :icon="Refresh"
          circle
          :loading="listRefreshing"
          @click="refreshFull"
        />
      </el-tooltip>
    </div>

    <div class="runs-table-wrap">
    <el-table
      v-loading="listLoading"
      :data="runs"
      stripe
      size="small"
      class="runs-table"
      @row-click="openRun"
    >
      <el-table-column label="State" width="88">
        <template #default="{ row }">
          <el-tag :type="stateTag(row.state)" size="small" :effect="row.state === 'running' ? 'dark' : 'light'">
            {{ row.state }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="Run" min-width="100" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.label || row.run_name || "—" }}
        </template>
      </el-table-column>
      <el-table-column prop="config_id" label="Config" min-width="120" show-overflow-tooltip class-name="col-config" />
      <el-table-column label="Progress" min-width="120" show-overflow-tooltip class-name="col-progress">
        <template #default="{ row }">
          <el-tag
            v-if="row.progress?.phase === 'waiting_disk_export'"
            type="warning"
            size="small"
            class="phase-tag"
          >
            Disk wait
          </el-tag>
          <template v-if="row.progress?.step != null">
            step {{ row.progress.step }}
            <template v-if="row.progress.max_steps">/ {{ row.progress.max_steps }}</template>
            <template v-if="row.progress.percent != null"> ({{ row.progress.percent }}%)</template>
            <span v-if="row.progress.loss != null" class="loss-cell">
              · {{ Number(row.progress.loss).toFixed(4) }}
            </span>
            <span v-if="formatRunProgressHint(row.progress)" class="loss-cell">
              · {{ formatRunProgressHint(row.progress) }}
            </span>
          </template>
          <span v-else-if="row.progress?.phase !== 'waiting_disk_export'">—</span>
        </template>
      </el-table-column>
      <el-table-column label="Updated" min-width="132" show-overflow-tooltip class-name="col-updated">
        <template #default="{ row }">
          {{ formatTime(row.progress?.updated_at || row.finished_at || row.started_at) }}
        </template>
      </el-table-column>
      <el-table-column label="" width="188" align="right" class-name="col-actions">
        <template #default="{ row }">
          <el-space class="row-actions" @click.stop>
            <el-tooltip content="Open run" placement="top">
              <el-button size="small" :icon="View" circle @click.stop="openRun(row)" />
            </el-tooltip>
            <el-tooltip v-if="row.config_id" content="Edit config" placement="top">
              <el-button size="small" :icon="Edit" circle @click.stop="goEditConfigId(row.config_id)" />
            </el-tooltip>
            <el-tooltip v-if="row.run_dir" content="Continue training" placement="top">
              <el-button size="small" :icon="VideoPlay" circle @click.stop="goContinue(row)" />
            </el-tooltip>
            <template v-if="row.state === 'pending'">
              <el-tooltip content="Run now" placement="top">
                <el-button size="small" :icon="VideoPlay" circle @click.stop="startQueuedNow(row.job_id)" />
              </el-tooltip>
              <el-tooltip content="Move up" placement="top">
                <el-button size="small" :icon="Top" circle @click.stop="move(row.job_id, 'up')" />
              </el-tooltip>
              <el-tooltip content="Move down" placement="top">
                <el-button size="small" :icon="Bottom" circle @click.stop="move(row.job_id, 'down')" />
              </el-tooltip>
              <el-tooltip content="Remove" placement="top">
                <el-button size="small" :icon="Delete" circle @click.stop="removeQueued(row.job_id)" />
              </el-tooltip>
            </template>
            <el-tooltip v-if="row.state === 'running' || row.state === 'stopping'" content="Stop" placement="top">
              <el-button size="small" :icon="VideoPause" circle @click.stop="stop(row.job_id)" />
            </el-tooltip>
          </el-space>
        </template>
      </el-table-column>
    </el-table>
    </div>

    <div v-if="runsTotal > listPageSize" class="runs-pagination">
      <el-pagination
        v-model:current-page="listPage"
        v-model:page-size="listPageSize"
        :total="runsTotal"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next"
        small
        background
        @current-change="loadRuns"
        @size-change="() => loadRuns(1)"
      />
    </div>

    <el-empty v-if="!listLoading && !runs.length" description="No training runs yet — start one above." />

    <el-dialog
      v-model="importOpen"
      title="Import training run folder"
      width="640px"
      destroy-on-close
      @closed="resetImportDialog"
    >
      <el-form label-position="top">
        <el-form-item label="Output directory (browse runs)">
          <PathFieldControl
            v-model="importOutputDir"
            placeholder="output"
            expect="dir"
            input-class="w-full"
            @change="loadImportCandidates"
          >
            <template #append>
              <el-button @click="loadImportCandidates">Scan</el-button>
            </template>
          </PathFieldControl>
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
          <PathFieldControl
            v-model="importRunPath"
            placeholder="output/20250217_14-30-00 or absolute path"
            expect="dir"
            required
            input-class="w-full"
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
          <PathFieldControl
            v-model="editForm.resume_from"
            placeholder="optional"
            expect="dir"
            input-class="w-full"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editOpen = false">Cancel</el-button>
        <el-button type="primary" @click="saveEdit">Save</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import {
  Bottom,
  CircleCheck,
  Delete,
  Edit,
  FolderOpened,
  Plus,
  Top,
  View,
  Refresh,
  VideoPause,
  VideoPlay,
} from "@element-plus/icons-vue";
import { api } from "../api";
import AutoRefreshBar from "../components/AutoRefreshBar.vue";
import TrainLivePanel from "../components/TrainLivePanel.vue";
import PathFieldControl from "../components/PathFieldControl.vue";
import { useAutoRefresh } from "../composables/useAutoRefresh";
import { useTrainLiveStream } from "../composables/useTrainLiveStream";
import { TRAIN_LIVE_REFRESH_STORAGE_KEY } from "../lib/autoRefresh";
import { formatError } from "../lib/formatError";
import { formatRunProgressHint } from "../lib/formatRunProgress";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";
import type {
  ConfigSearchItem,
  ImportCandidatesResult,
  ImportRunPreview,
  JobRecord,
  JobStartBody,
  TrainingRunRow,
} from "../types/api";

const router = useRouter();
const vLoading = ElLoadingDirective;

const runs = ref<TrainingRunRow[]>([]);
const runsTotal = ref(0);
const listPage = ref(1);
const listPageSize = ref(20);
const listQuery = ref("");
const stateFilter = ref("");
const listLoading = ref(false);
const listRefreshing = ref(false);
const activeRun = ref<TrainingRunRow | null>(null);
const stats = ref({ running: 0, pending: 0 });
const hasAnyConfig = ref(false);
const configOptions = ref<ConfigSearchItem[]>([]);
const configId = ref<string | null>(getJobConfigId());
const configValid = ref<boolean | null>(null);
const validating = ref(false);
const numGpus = ref(1);
const resumeFrom = ref("");
const trustCache = ref(false);
const regenerateCache = ref(false);
const error = ref("");
const pollWarning = ref("");

const importOpen = ref(false);
const importRunPath = ref("");
const importOutputDir = ref("output");
const importCandidates = ref<NonNullable<ImportCandidatesResult["runs"]>>([]);
const importPreview = ref<ImportRunPreview | null>(null);
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
  runs.value.filter((j) => j.state === "running" || j.state === "stopping")
);

const canLaunch = computed(() => Boolean(configId.value && hasAnyConfig.value));

function stateTag(state: string | undefined): "primary" | "success" | "warning" | "info" | "danger" {
  if (state === "running" || state === "stopping") return "success";
  if (state === "pending") return "warning";
  if (state === "finished") return "info";
  if (state === "on_disk") return "primary";
  if (state === "stopped") return "warning";
  if (state === "failed") return "danger";
  return "info";
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return String(iso).slice(0, 19).replace("T", " ");
}

function runFolderLabel(row: TrainingRunRow): string {
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

async function loadImportCandidates(): Promise<void> {
  const data = await api.listImportCandidates(importOutputDir.value || "output");
  importCandidates.value = data.runs || [];
}

function onImportPathPicked(path: string): void {
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
      importForm.config_id = String(data.suggested_config_id || "");
    }
    if (!importForm.dataset_id) {
      importForm.dataset_id = String(data.suggested_dataset_id || "");
    }
  } catch (e) {
    ElMessage.error(formatError(e));
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
    await refreshFull();
    router.push({ name: "job-detail", params: { id: job.id } });
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    importSaving.value = false;
  }
}

async function refreshConfigAvailability(): Promise<void> {
  try {
    const r = await api.searchConfigs({ q: "", page: 1, page_size: 1 });
    hasAnyConfig.value = (r.total ?? 0) > 0;
  } catch {
    hasAnyConfig.value = false;
  }
}

async function searchConfigOptions(query: string): Promise<void> {
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

async function loadRuns(page = listPage.value): Promise<void> {
  listLoading.value = true;
  listPage.value = page;
  try {
    const data = await api.trainRuns({
      page: listPage.value,
      page_size: listPageSize.value,
      q: listQuery.value.trim(),
      state: stateFilter.value || "",
      include_disk: true,
    });
    runs.value = data.items || [];
    runsTotal.value = data.total ?? 0;
    stats.value = data.stats || { running: 0, pending: 0 };
  } catch (e) {
    error.value = formatError(e);
    runs.value = [];
  } finally {
    listLoading.value = false;
  }
}

async function refreshActive(): Promise<boolean> {
  try {
    const data = await api.trainActive();
    activeRun.value = data.active || null;
    pollWarning.value = "";
    return true;
  } catch (e) {
    pollWarning.value = formatError(e);
    return false;
  }
}

async function refreshFull(): Promise<void> {
  listRefreshing.value = true;
  try {
    await Promise.all([loadRuns(listPage.value), refreshActive(), refreshConfigAvailability()]);
    syncConfigSelection();
    pollWarning.value = "";
  } catch (e) {
    pollWarning.value = formatError(e);
  } finally {
    listRefreshing.value = false;
  }
}

const hasLiveRun = computed(
  () => !!activeRun.value || (stats.value.running ?? 0) > 0
);

const activeJobId = computed(() => {
  const id = activeRun.value?.job_id;
  return id != null && id !== "" ? String(id) : "";
});

const liveStream = useTrainLiveStream(activeJobId, {
  onRunFinished: () => {
    void refreshActive().then((ok) => {
      if (ok && !activeRun.value) void loadRuns(listPage.value);
    });
  },
});

const {
  logText: liveLogText,
  streamStatus: liveStreamStatus,
  streamError: liveStreamError,
  useHttpFallback: liveUseHttpFallback,
  progress: liveProgress,
  scalars: liveScalars,
  previewImages: livePreviewImages,
} = liveStream;

function mergeLiveIntoActiveRun(): void {
  const row = activeRun.value;
  if (!row) return;
  const next = { ...row };
  if (liveProgress.value) {
    next.progress = liveProgress.value;
  }
  if (Object.keys(liveScalars.value).length) {
    next.scalars = liveScalars.value;
  }
  if (livePreviewImages.value.length) {
    next.preview_images = livePreviewImages.value;
  }
  activeRun.value = next;
}

watch(
  () => [liveProgress.value, liveScalars.value, livePreviewImages.value] as const,
  () => {
    if (activeRun.value && !liveUseHttpFallback.value) {
      mergeLiveIntoActiveRun();
    }
  },
  { deep: true }
);

watch(activeJobId, (id, prev) => {
  if (id && id !== prev) {
    void refreshActive();
  }
});

const {
  intervalSec: liveIntervalSec,
  isLoading: liveMetricsLoading,
  refreshing: liveRefreshing,
  polling: livePolling,
  lastUpdated: liveLastUpdated,
  paused: livePaused,
  setIntervalSec: setLiveInterval,
  refreshNow: refreshLiveNow,
} = useAutoRefresh({
  storageKey: TRAIN_LIVE_REFRESH_STORAGE_KEY,
  immediate: false,
  refresh: async (signal) => {
    const hadActive = !!activeRun.value;
    const ok = await refreshActive();
    if (signal?.aborted) return;
    if (!ok) return;
    if (activeRun.value && !liveUseHttpFallback.value && liveProgress.value) {
      mergeLiveIntoActiveRun();
    }
    if (hadActive && !activeRun.value) {
      await loadRuns(listPage.value);
    }
  },
  isActive: () => hasLiveRun.value && liveUseHttpFallback.value,
});

onMounted(() => {
  void refreshFull();
});

function goPickConfig() {
  router.push({ name: "configs-list", query: { pick: "job" } });
}

function goEditConfig() {
  if (!configId.value) return;
  router.push({ name: "configs-detail", params: { configId: String(configId.value) } });
}

function goCreateConfig() {
  router.push({ name: "configs-new" });
}

async function checkConfig(): Promise<void> {
  if (!configId.value) return;
  validating.value = true;
  configValid.value = null;
  try {
    const cfg = await api.getConfig(configId.value);
    const r = await api.validate(cfg.content);
    configValid.value = !!r.ok;
    if (!r.ok) {
      ElMessage.warning(r.error || "Config is not valid yet");
    }
  } catch (e) {
    configValid.value = false;
    ElMessage.error(formatError(e));
  } finally {
    validating.value = false;
  }
}

watch(configId, (id, prev) => {
  if (id && id !== prev) {
    void checkConfig();
  }
});

watch(regenerateCache, (on) => {
  if (on) trustCache.value = false;
});

function goJob(id: string) {
  router.push({ name: "job-detail", params: { id } });
}

function openRun(row: TrainingRunRow) {
  if (!row) return;
  if (row.kind === "job" && row.job_id != null) {
    goJob(row.job_id);
    return;
  }
  if (row.run_name) {
    router.push({ name: "run-detail", params: { name: row.run_name } });
  }
}

function goEditConfigId(id: string | number | null | undefined) {
  if (!id) return;
  router.push({ name: "configs-detail", params: { configId: String(id) } });
}

function goContinue(row: TrainingRunRow) {
  if (!row?.run_dir) return;
  router.push({ name: "configs-new", query: { continue_run: row.run_dir } });
}

function trainingJobBody(cacheOnly: boolean): JobStartBody | null {
  const selectedConfigId = configId.value;
  if (!selectedConfigId) return null;
  return {
    config_id: selectedConfigId,
    num_gpus: numGpus.value,
    resume_from: cacheOnly ? undefined : resumeFrom.value || undefined,
    cache_only: cacheOnly,
    trust_cache: cacheOnly ? false : trustCache.value,
    regenerate_cache: regenerateCache.value,
    enqueue: !cacheOnly,
    start_immediately: false,
  };
}

async function submitJob(body: JobStartBody, successMessage: string) {
  error.value = "";
  setJobConfigId(String(body.config_id));
  try {
    await api.startJob(body);
    ElMessage.success(successMessage);
    await refreshFull();
  } catch (e) {
    error.value = formatError(e);
    ElMessage.error(formatError(e));
  }
}

async function enqueue() {
  if (!canLaunch.value) return;
  const body = trainingJobBody(false);
  if (!body) return;
  await submitJob({ ...body, enqueue: true, start_immediately: false }, "Added to queue");
}

async function startNow() {
  if (!canLaunch.value) return;
  const body = trainingJobBody(false);
  if (!body) return;
  await submitJob(
    { ...body, enqueue: false, start_immediately: true },
    "Job started or queued at front"
  );
}

async function enqueueCache() {
  if (!canLaunch.value) return;
  const body = trainingJobBody(true);
  if (!body) return;
  await submitJob({ ...body, enqueue: true, start_immediately: false }, "Cache build queued");
}

async function startCacheNow() {
  if (!canLaunch.value) return;
  const body = trainingJobBody(true);
  if (!body) return;
  await submitJob(
    { ...body, enqueue: false, start_immediately: true },
    "Cache build started or queued at front"
  );
}

async function stop(id: string | null | undefined) {
  if (!id) return;
  await api.stopJob(id);
  ElMessage.info("Stop requested");
  await refreshFull();
}

async function sendRunSignal(
  row: { job_id?: string | null },
  type: string
) {
  const id = row?.job_id;
  if (!id) return;
  try {
    await api.sendJobSignal(id, type);
    ElMessage.success(`Signal "${type}" sent`);
    await refreshFull();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function move(id: string | null | undefined, direction: "up" | "down") {
  if (!id) return;
  await api.moveJobQueue(id, direction);
  await refreshFull();
}

async function startQueuedNow(id: string | null | undefined) {
  if (!id) return;
  await api.startJobNow(id);
  ElMessage.success("Moved to front");
  await refreshFull();
}

async function removeQueued(id: string | null | undefined) {
  if (!id) return;
  await ElMessageBox.confirm("Remove this job from the queue?", "Confirm", { type: "warning" });
  await api.deleteJob(id);
  ElMessage.success("Removed");
  await refreshFull();
}

function openEdit(row: JobRecord) {
  editJobId.value = row.id;
  editForm.config_id = row.config_id != null ? String(row.config_id) : "";
  editForm.num_gpus = row.num_gpus || 1;
  editForm.resume_from = row.resume_from || "";
  searchConfigOptions(String(row.config_id || ""));
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
  await refreshFull();
}
</script>

<style scoped>
.jobs-page {
  max-width: 100%;
}
.stats-line {
  display: flex;
  gap: var(--rf-space-sm);
}
.stat-running {
  color: var(--el-color-success);
  font-weight: 600;
}
.launch-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--rf-space-sm);
  flex-wrap: wrap;
  font-weight: 600;
}
.launch-form {
  display: flex;
  flex-direction: column;
  gap: var(--rf-space-sm);
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
  gap: var(--rf-space-xs);
  margin-bottom: var(--rf-space-xs);
}
.hint {
  display: block;
}
.block-hint {
  margin-top: 4px;
}
.cache-options :deep(.el-form-item__content) {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}
.cache-option-second {
  margin-left: 0;
}
.config-block--compact {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--rf-space-sm);
}
.config-block--compact .page-hint {
  margin: 0;
}
.page-panel {
  padding: 14px;
}
.config-selected-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--rf-space-xs);
  margin-bottom: var(--rf-space-xs);
}
.config-id {
  font-family: var(--rf-font-mono);
  font-size: 13px;
}
.runs-table-wrap {
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
}
.runs-table {
  width: 100%;
  cursor: pointer;
}
@media (max-width: 960px) {
  .runs-table :deep(.col-updated),
  .runs-table :deep(.col-config) {
    display: none;
  }
}
.loss-cell {
  font-family: var(--rf-font-mono);
  font-size: 12px;
}
.runs-pagination {
  margin-top: var(--rf-space-sm);
  display: flex;
  justify-content: flex-end;
}
.row-actions {
  justify-content: flex-end;
}
.mt-8 {
  margin-top: var(--rf-space-xs);
}
.mt-12 {
  margin-top: var(--rf-space-sm);
}
</style>

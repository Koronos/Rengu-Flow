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
      <el-space wrap>
        <el-button type="primary" :icon="Plus" @click="newRun">New run</el-button>
        <el-button :icon="FolderOpened" @click="openImportDialog">Import existing run</el-button>
      </el-space>
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

    <div class="page-toolbar">
      <el-input
        v-model="listQuery"
        clearable
        placeholder="Search run folder, name…"
        class="page-toolbar-search"
        @input="loadRuns"
        @clear="loadRuns"
      />
      <el-tooltip content="Refresh" :show-after="300">
        <el-button size="small" :icon="Refresh" circle :loading="listRefreshing" @click="refreshFull" />
      </el-tooltip>
    </div>

    <!-- Saved drafts -->
    <el-card v-if="savedRuns.length" shadow="never" class="page-section">
      <template #header><span>Saved</span></template>
      <div class="run-rows">
        <div v-for="row in savedRuns" :key="row.key" class="run-row">
          <div class="run-row__main">
            <el-tag :type="stateTag(row.state)" size="small">new</el-tag>
            <span class="run-row__name">{{ row.label || row.run_name || "—" }}</span>
          </div>
          <el-space class="run-row__actions">
            <el-button size="small" type="primary" :icon="Plus" @click="addToQueue(row.job_id)">
              Add to queue
            </el-button>
            <el-button size="small" :icon="Edit" @click="editRun(row)">Edit</el-button>
            <el-button size="small" :icon="CopyDocument" @click="newRunFromConfig(row.job_id)">
              New from config
            </el-button>
            <el-button size="small" :icon="Delete" @click="removeRun(row)">Delete</el-button>
          </el-space>
        </div>
      </div>
    </el-card>

    <!-- Queue: running (pinned) + pending (drag & drop) -->
    <el-card shadow="never" class="page-section">
      <template #header><span>Queue</span></template>
      <el-empty
        v-if="!runningRuns.length && !pendingRuns.length"
        description="Nothing running or queued."
        :image-size="56"
      />
      <div v-else class="run-rows">
        <div v-for="row in runningRuns" :key="row.key" class="run-row run-row--active">
          <div class="run-row__main">
            <el-tag :type="stateTag(row.state)" size="small" effect="dark">{{ row.state }}</el-tag>
            <span class="run-row__name">{{ row.label || row.run_name || "—" }}</span>
            <span v-if="row.progress?.step != null" class="run-row__progress">
              step {{ row.progress.step }}<template v-if="row.progress.max_steps">/{{ row.progress.max_steps }}</template>
            </span>
          </div>
          <el-space class="run-row__actions">
            <el-button size="small" :icon="View" @click="openRun(row)">Open</el-button>
            <el-button size="small" :icon="VideoPause" @click="stop(row.job_id)">Stop</el-button>
          </el-space>
        </div>

        <div v-if="runningRuns.length && pendingRuns.length" class="queue-sep" />

        <div ref="pendingListEl" class="run-rows run-rows--pending">
          <div
            v-for="row in pendingRuns"
            :key="row.key"
            class="run-row run-row--pending"
          >
            <div class="run-row__main">
              <el-icon class="drag-handle"><Rank /></el-icon>
              <el-tag :type="stateTag(row.state)" size="small">queued</el-tag>
              <span class="run-row__name">{{ row.label || row.run_name || "—" }}</span>
            </div>
            <el-space class="run-row__actions">
              <el-button size="small" :icon="VideoPlay" @click="startQueuedNow(row.job_id)">Run now</el-button>
              <el-button size="small" :icon="Edit" @click="editRun(row)">Edit</el-button>
              <el-button size="small" :icon="CopyDocument" @click="newRunFromConfig(row.job_id)">
                New from config
              </el-button>
              <el-button size="small" :icon="Delete" @click="removeRun(row)">Delete</el-button>
            </el-space>
          </div>
        </div>
      </div>
    </el-card>

    <!-- History -->
    <el-card shadow="never" class="page-section">
      <template #header>
        <div class="history-header">
          <span class="history-title">History</span>
          <el-input
            v-model="historyQuery"
            size="small"
            clearable
            placeholder="Search history…"
            class="history-search"
            :prefix-icon="Search"
          />
          <el-select
            v-model="historyState"
            size="small"
            clearable
            placeholder="All states"
            class="history-filter"
          >
            <el-option label="All" value="" />
            <el-option label="Finished" value="finished" />
            <el-option label="Stopped" value="stopped" />
            <el-option label="Error" value="failed" />
          </el-select>
        </div>
      </template>
      <el-table
        v-loading="listLoading"
        :data="historyRuns"
        stripe
        size="small"
        class="runs-table"
        :default-sort="{ prop: 'updated', order: 'descending' }"
        @row-click="openRun"
      >
        <el-table-column
          label="State"
          width="92"
          prop="state"
          sortable
          :sort-method="sortHistoryState"
        >
          <template #default="{ row }">
            <el-tag :type="stateTag(row.state)" size="small">{{ stateLabel(row.state) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column
          label="Run"
          min-width="140"
          show-overflow-tooltip
          prop="run"
          sortable
          :sort-method="sortHistoryRun"
        >
          <template #default="{ row }">{{ row.label || row.run_name || "—" }}</template>
        </el-table-column>
        <el-table-column
          label="Progress"
          min-width="140"
          show-overflow-tooltip
          prop="progress"
          sortable
          :sort-method="sortHistoryProgress"
        >
          <template #default="{ row }">
            <template v-if="row.progress?.step != null">
              step {{ row.progress.step }}
              <template v-if="row.progress.max_steps">/ {{ row.progress.max_steps }}</template>
              <span v-if="row.progress.loss != null" class="loss-cell">
                · {{ Number(row.progress.loss).toFixed(4) }}
              </span>
            </template>
            <span v-else>—</span>
          </template>
        </el-table-column>
        <el-table-column
          label="Updated"
          min-width="150"
          show-overflow-tooltip
          prop="updated"
          sortable
          :sort-method="sortHistoryUpdated"
        >
          <template #default="{ row }">
            {{ formatTime(row.progress?.updated_at || row.finished_at || row.started_at) }}
          </template>
        </el-table-column>
        <el-table-column label="" width="150" align="right">
          <template #default="{ row }">
            <el-dropdown v-if="isMobile" trigger="click" @click.stop>
              <el-button size="small" circle :icon="MoreFilled" @click.stop />
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item :icon="View" @click="openRun(row as TrainingRunRow)">
                    Open
                  </el-dropdown-item>
                  <el-dropdown-item
                    v-if="row.run_dir"
                    :icon="VideoPlay"
                    @click="continueRun(row as TrainingRunRow)"
                  >
                    Continue training
                  </el-dropdown-item>
                  <el-dropdown-item
                    v-if="row.job_id"
                    :icon="CopyDocument"
                    @click="newRunFromConfig(row.job_id)"
                  >
                    New run from this config
                  </el-dropdown-item>
                  <el-dropdown-item :icon="Delete" @click="removeRun(row as TrainingRunRow)">
                    Delete from list
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
            <el-space v-else class="row-actions" @click.stop>
              <el-tooltip content="Open" :show-after="300">
                <el-button
                  size="small"
                  circle
                  :icon="View"
                  @click.stop="openRun(row as TrainingRunRow)"
                />
              </el-tooltip>
              <el-tooltip v-if="row.run_dir" content="Continue training" :show-after="300">
                <el-button
                  size="small"
                  circle
                  :icon="VideoPlay"
                  @click.stop="continueRun(row as TrainingRunRow)"
                />
              </el-tooltip>
              <el-tooltip v-if="row.job_id" content="New run from this config" :show-after="300">
                <el-button
                  size="small"
                  circle
                  :icon="CopyDocument"
                  @click.stop="newRunFromConfig(row.job_id)"
                />
              </el-tooltip>
              <el-tooltip content="Delete from list" :show-after="300">
                <el-button
                  size="small"
                  circle
                  :icon="Delete"
                  @click.stop="removeRun(row as TrainingRunRow)"
                />
              </el-tooltip>
            </el-space>
          </template>
        </el-table-column>
      </el-table>
      <el-empty
        v-if="!listLoading && !historyRuns.length"
        description="No finished runs yet."
        :image-size="56"
      />
    </el-card>

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
        <el-text v-if="importPreviewLoading" type="info" size="small">Inspecting run folder…</el-text>
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
        <el-descriptions-item label="Config">{{ importPreview.config_path || "—" }}</el-descriptions-item>
        <el-descriptions-item label="Artifacts">
          {{ importPreview.run.artifacts?.length || 0 }}
        </el-descriptions-item>
        <el-descriptions-item label="TensorBoard">
          {{ importPreview.run.has_tensorboard ? "yes" : "no" }}
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />
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
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import {
  CopyDocument,
  Delete,
  Edit,
  FolderOpened,
  MoreFilled,
  Plus,
  Rank,
  Refresh,
  Search,
  View,
  VideoPause,
  VideoPlay,
} from "@element-plus/icons-vue";
import Sortable from "sortablejs";
import { api } from "../api";
import AutoRefreshBar from "../components/AutoRefreshBar.vue";
import TrainLivePanel from "../components/TrainLivePanel.vue";
import PathFieldControl from "../components/PathFieldControl.vue";
import { useAutoRefresh } from "../composables/useAutoRefresh";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useTrainLiveStream } from "../composables/useTrainLiveStream";
import { TRAIN_LIVE_REFRESH_STORAGE_KEY } from "../lib/autoRefresh";
import { formatError } from "../lib/formatError";
import { useConfigEditorStore } from "../stores/configEditor";
import type { ImportCandidatesResult, ImportRunPreview, TrainingRunRow } from "../types/api";

const router = useRouter();
const vLoading = ElLoadingDirective;
const editor = useConfigEditorStore();
const { isMobile } = useBreakpoint();

const runs = ref<TrainingRunRow[]>([]);
const listQuery = ref("");
const listLoading = ref(false);
const listRefreshing = ref(false);
const activeRun = ref<TrainingRunRow | null>(null);
const stats = ref({ running: 0, pending: 0 });
const error = ref("");
const pollWarning = ref("");

const importOpen = ref(false);
const importRunPath = ref("");
const importOutputDir = ref("output");
const importCandidates = ref<NonNullable<ImportCandidatesResult["runs"]>>([]);
const importPreview = ref<ImportRunPreview | null>(null);
const importPreviewLoading = ref(false);
const importSaving = ref(false);
const importForm = reactive({ import_dataset: true, dataset_id: "" });
let importPreviewTimer: ReturnType<typeof setTimeout> | null = null;

const pendingListEl = ref<HTMLElement | null>(null);
let sortable: Sortable | null = null;

const savedRuns = computed(() => runs.value.filter((r) => r.state === "new"));
const runningRuns = computed(() =>
  runs.value.filter((r) => r.state === "running" || r.state === "stopping")
);
const pendingRuns = computed(() => runs.value.filter((r) => r.state === "pending"));
const historyState = ref("");
const historyQuery = ref("");
const historyRuns = computed(() => {
  const q = historyQuery.value.trim().toLowerCase();
  return runs.value.filter((r) => {
    if (!["finished", "stopped", "failed"].includes(String(r.state))) return false;
    if (historyState.value && String(r.state) !== historyState.value) return false;
    if (q) {
      const haystack = `${r.run_name ?? ""} ${r.label ?? ""} ${r.run_dir ?? ""}`.toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    return true;
  });
});

function historyUpdatedKey(r: TrainingRunRow): string {
  return String(r.progress?.updated_at || r.finished_at || r.started_at || "");
}
function sortHistoryUpdated(a: TrainingRunRow, b: TrainingRunRow): number {
  return historyUpdatedKey(a).localeCompare(historyUpdatedKey(b));
}
function sortHistoryRun(a: TrainingRunRow, b: TrainingRunRow): number {
  const ka = (a.label || a.run_name || "").toLowerCase();
  const kb = (b.label || b.run_name || "").toLowerCase();
  return ka.localeCompare(kb);
}
function sortHistoryState(a: TrainingRunRow, b: TrainingRunRow): number {
  return stateLabel(a.state).localeCompare(stateLabel(b.state));
}
function sortHistoryProgress(a: TrainingRunRow, b: TrainingRunRow): number {
  return (a.progress?.step ?? -1) - (b.progress?.step ?? -1);
}

function stateTag(state: string | undefined): "primary" | "success" | "warning" | "info" | "danger" {
  if (state === "running" || state === "stopping") return "success";
  if (state === "pending") return "warning";
  if (state === "new") return "info";
  if (state === "finished") return "info";
  if (state === "stopped") return "warning";
  if (state === "failed") return "danger";
  return "info";
}

function stateLabel(state: string | undefined): string {
  if (state === "finished") return "Finished";
  if (state === "stopped") return "Stopped";
  if (state === "failed") return "Error";
  return String(state ?? "—");
}

function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return String(iso).slice(0, 19).replace("T", " ");
}

// --- Import dialog ---

function openImportDialog() {
  importOpen.value = true;
  loadImportCandidates().catch(() => {});
}

function resetImportDialog() {
  if (importPreviewTimer) {
    clearTimeout(importPreviewTimer);
    importPreviewTimer = null;
  }
  importRunPath.value = "";
  importPreview.value = null;
  importForm.import_dataset = true;
  importForm.dataset_id = "";
}

async function loadImportCandidates(): Promise<void> {
  const data = await api.listImportCandidates(importOutputDir.value || "output");
  importCandidates.value = data.runs || [];
}

// Auto-inspect the run folder shortly after the path changes (debounced so we don't
// hit the backend on every keystroke). Picking from the dropdown updates the same v-model.
watch(importRunPath, (path) => {
  if (importPreviewTimer) clearTimeout(importPreviewTimer);
  if (!(path || "").trim()) {
    importPreview.value = null;
    return;
  }
  importPreviewTimer = setTimeout(() => void previewImportRun({ silent: true }), 500);
});

async function previewImportRun(opts: { silent?: boolean } = {}): Promise<void> {
  const path = importRunPath.value.trim();
  if (!path) {
    importPreview.value = null;
    return;
  }
  importPreviewLoading.value = true;
  importPreview.value = null;
  try {
    const data = await api.previewJobImport(path);
    importPreview.value = data;
    if (!importForm.dataset_id) importForm.dataset_id = String(data.suggested_dataset_id || "");
  } catch (e) {
    importPreview.value = null;
    // While typing, a partial/invalid path is expected — don't spam error toasts.
    if (!opts.silent) ElMessage.error(formatError(e));
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

// --- Runs list ---

async function loadRuns(): Promise<void> {
  listLoading.value = true;
  try {
    const data = await api.trainRuns({
      page: 1,
      page_size: 100,
      q: listQuery.value.trim(),
    });
    runs.value = data.items || [];
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
    await Promise.all([loadRuns(), refreshActive()]);
    pollWarning.value = "";
  } catch (e) {
    pollWarning.value = formatError(e);
  } finally {
    listRefreshing.value = false;
  }
}

const hasLiveRun = computed(() => !!activeRun.value || (stats.value.running ?? 0) > 0);

const activeJobId = computed(() => {
  const id = activeRun.value?.job_id;
  return id != null && id !== "" ? String(id) : "";
});

const liveStream = useTrainLiveStream(activeJobId, {
  onRunFinished: () => {
    void refreshActive().then((ok) => {
      if (ok && !activeRun.value) void loadRuns();
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
  if (liveProgress.value) next.progress = liveProgress.value;
  if (Object.keys(liveScalars.value).length) next.scalars = liveScalars.value;
  if (livePreviewImages.value.length) next.preview_images = livePreviewImages.value;
  activeRun.value = next;
}

watch(
  () => [liveProgress.value, liveScalars.value, livePreviewImages.value] as const,
  () => {
    if (activeRun.value && !liveUseHttpFallback.value) mergeLiveIntoActiveRun();
  },
  { deep: true }
);

watch(activeJobId, (id, prev) => {
  if (id && id !== prev) void refreshActive();
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
    if (hadActive && !activeRun.value) await loadRuns();
  },
  isActive: () => hasLiveRun.value && liveUseHttpFallback.value,
});

// --- Drag & drop reordering of the pending queue ---

function setupSortable(): void {
  if (sortable || !pendingListEl.value) return;
  sortable = Sortable.create(pendingListEl.value, {
    handle: ".drag-handle",
    animation: 150,
    onEnd: (evt) => {
      const ids = pendingRuns.value.map((r) => r.job_id).filter(Boolean) as (string | number)[];
      if (evt.oldIndex == null || evt.newIndex == null) return;
      const [moved] = ids.splice(evt.oldIndex, 1);
      ids.splice(evt.newIndex, 0, moved);
      void api
        .reorderQueue(ids)
        .then(refreshFull)
        .catch((e) => ElMessage.error(formatError(e)));
    },
  });
}

watch(
  () => pendingRuns.value.length,
  async (n) => {
    await nextTick();
    if (n > 0) setupSortable();
    else {
      sortable?.destroy();
      sortable = null;
    }
  }
);

onMounted(() => {
  void refreshFull();
});

onBeforeUnmount(() => {
  sortable?.destroy();
  sortable = null;
});

// --- Navigation / row actions ---

function goJob(id: string) {
  router.push({ name: "job-detail", params: { id } });
}

function openRun(row: TrainingRunRow) {
  if (!row) return;
  if (row.kind === "job" && row.job_id != null) {
    goJob(String(row.job_id));
    return;
  }
  if (row.run_name) router.push({ name: "run-detail", params: { name: row.run_name } });
}

async function stop(id: string | null | undefined) {
  if (!id) return;
  await api.stopJob(String(id));
  ElMessage.info("Stop requested");
  await refreshFull();
}

async function sendRunSignal(row: { job_id?: string | null }, type: string) {
  const id = row?.job_id;
  if (!id) return;
  try {
    await api.sendJobSignal(String(id), type);
    ElMessage.success(`Signal "${type}" sent`);
    await refreshFull();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function startQueuedNow(id: string | null | undefined) {
  if (!id) return;
  await api.startJobNow(String(id));
  ElMessage.success("Moved to front");
  await refreshFull();
}

async function addToQueue(id: string | null | undefined) {
  if (!id) return;
  try {
    await api.enqueueJob(String(id));
    ElMessage.success("Added to queue");
    await refreshFull();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

async function removeRun(row: TrainingRunRow) {
  if (!row?.job_id) return;
  await ElMessageBox.confirm(
    "Removes this run from the list only. Files on disk (checkpoints, logs, samples) are NOT deleted.",
    "Delete run",
    { type: "warning", confirmButtonText: "Delete", cancelButtonText: "Cancel" }
  );
  try {
    await api.deleteJob(String(row.job_id));
    ElMessage.success("Removed from list");
    await refreshFull();
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

// --- Run form (a dedicated section/page, not a modal) ---

async function newRun() {
  // Prepare a blank run in the shared store, then open the form section.
  await editor.fetchSchema();
  await editor.newConfig();
  router.push({ name: "run-new" });
}

async function newRunFromConfig(id: string | null | undefined) {
  if (!id) return;
  try {
    const { content } = await api.seedJobConfig(String(id));
    await editor.fetchSchema();
    await editor.loadContent(content);
    router.push({ name: "run-new" });
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

function editRun(row: TrainingRunRow) {
  if (!row?.job_id) return;
  router.push({ name: "run-edit", params: { id: String(row.job_id) } });
}

function continueRun(row: TrainingRunRow) {
  if (!row?.run_dir) return;
  router.push({ name: "run-new", query: { continue_run: row.run_dir } });
}
</script>

<style scoped>
.jobs-page {
  max-width: 100%;
}
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
.page-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: var(--rf-space-md);
}
.page-toolbar-search {
  max-width: 320px;
}
.run-rows {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.run-rows--pending {
  margin-top: 0;
}
.run-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 12px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  background: var(--el-bg-color);
  flex-wrap: wrap;
}
.run-row--active {
  border-color: var(--el-color-success-light-5);
  background: var(--el-color-success-light-9);
}
.run-row--pending {
  cursor: default;
}
.run-row__main {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  flex: 1;
}
.run-row__name {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.run-row__progress {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.drag-handle {
  cursor: grab;
  color: var(--el-text-color-secondary);
}
.queue-sep {
  height: 1px;
  background: var(--el-border-color);
  margin: 4px 0;
}
.loss-cell {
  color: var(--el-text-color-secondary);
}
.runs-table {
  width: 100%;
}
.history-header {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.history-title {
  margin-right: auto;
}
.history-search {
  max-width: 220px;
}
.history-filter {
  max-width: 160px;
}
.mb-12 {
  margin-bottom: 12px;
}
.mt-12 {
  margin-top: 12px;
}
.mt-8 {
  margin-top: 8px;
}
.w-full {
  width: 100%;
}
</style>

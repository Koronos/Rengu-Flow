<template>
  <div class="run-comparison">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Overlay metrics and compare hyperparameters across runs</p>
        <el-text v-if="allRuns.length" type="info" size="small" class="page-head-meta">
          {{ selectedIds.length }}/{{ allRuns.length }} selected in “{{ outputDir }}”
        </el-text>
      </div>
    </div>

    <form class="folder-bar" @submit.prevent="applyFolder">
      <el-input
        v-model="folderInput"
        placeholder="output folder — e.g. output, runs/cosmos, or /abs/path"
        size="small"
        clearable
      >
        <template #prepend>Folder</template>
        <template #append><el-button native-type="submit">Load</el-button></template>
      </el-input>
    </form>

    <!-- Reload + opt-in auto-update for the whole comparison (run statuses, last scalars, and the
         overlay series), mirroring the loss monitor's controls on the run detail page. -->
    <div v-if="allRuns.length" class="cmp-refresh-bar">
      <el-button size="small" :icon="Refresh" :loading="refreshing" @click="reload">Reload</el-button>
      <el-checkbox v-model="autoUpdate" size="small">Auto-update</el-checkbox>
      <span class="cmp-cadence" :class="{ 'cmp-cadence--off': !autoUpdate }">
        every
        <el-input-number
          v-model="cadenceSec"
          :min="1"
          :max="3600"
          :step="1"
          :controls="false"
          :disabled="!autoUpdate"
          size="small"
          class="cmp-cadence-input"
        />
        s
      </span>
    </div>

    <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
    <div v-else-if="loading" class="loading"><el-text type="info">Loading runs…</el-text></div>
    <el-empty
      v-else-if="!allRuns.length"
      :description="`No comparable runs found in “${outputDir}”. Runs need a config or TensorBoard events.`"
    />

    <div v-else class="cmp-layout">
      <!-- Sidebar: native run selector (light-weight, no per-item EP components) -->
      <aside class="cmp-sidebar">
        <input v-model="search" class="cmp-search" type="search" placeholder="Filter runs…" />
        <div class="cmp-sidebar__sort">
          <select v-model="sortKey" class="cmp-sort-select" aria-label="Sort runs by">
            <option value="name">Name</option>
            <option value="updated">Recent</option>
            <option value="created">Created</option>
            <option value="loss">Loss</option>
            <option value="status">Status</option>
          </select>
          <button
            type="button"
            class="cmp-sort-dir"
            :title="sortDir === 'asc' ? 'Ascending — click for descending' : 'Descending — click for ascending'"
            @click="toggleSortDir"
          >
            {{ sortDir === "asc" ? "↑" : "↓" }}
          </button>
        </div>
        <div class="cmp-sidebar__actions">
          <button type="button" @click="selectRecent(4)">Recent 4</button>
          <button type="button" @click="selectAllVisible">All</button>
          <button type="button" @click="clearSelection">Clear</button>
        </div>
        <div class="cmp-runlist">
          <label v-for="r in sidebarRuns" :key="r.run_id" class="cmp-run">
            <input
              type="checkbox"
              :checked="selectedIds.includes(r.run_id)"
              @change="toggleRun(r.run_id)"
            />
            <span class="cmp-run__swatch" :style="{ background: colorMap[r.run_id] }"></span>
            <span class="cmp-run__body">
              <span class="cmp-run__name" :title="r.name">{{ r.name }}</span>
              <span class="cmp-run__meta">
                <span class="cmp-run__status" :class="`is-${r.status}`">{{ r.status }}</span>
                <span v-if="r.last_scalars && r.last_scalars['train/loss'] != null" class="cmp-run__loss">
                  loss {{ fmt(r.last_scalars['train/loss']) }}
                </span>
              </span>
            </span>
          </label>
          <div v-if="!sidebarRuns.length" class="cmp-runlist__empty">No runs match.</div>
        </div>
      </aside>

      <!-- Content -->
      <section class="cmp-content">
        <div v-if="!selectedIds.length" class="cmp-placeholder">
          <el-text type="info">Select runs from the left to overlay and compare them.</el-text>
        </div>

        <template v-else>
          <!-- Toolbar: native controls (slider re-renders charts, so keep it cheap) -->
          <div class="cmp-toolbar">
            <label class="cmp-ctrl">
              Smoothing
              <input v-model.number="smoothing" type="range" min="0" max="0.99" step="0.01" />
              <span class="cmp-ctrl__val">{{ smoothing.toFixed(2) }}</span>
            </label>
            <label class="cmp-ctrl"><input v-model="logScale" type="checkbox" /> Log scale (y)</label>
          </div>

          <!-- Overlay charts. Zoom + cursor are synced across every board, so their controls are
               global and live in one sticky bar that overlaps the top-right as you scroll. -->
          <div v-if="chartMetrics.length" class="cmp-boards">
            <div v-show="anyZoomed || pinnedStep != null" class="cmp-board-tools">
              <button
                v-if="pinnedStep != null"
                type="button"
                class="cmp-board-btn cmp-board-btn--unpin"
                @click="unpin"
              >
                📌 Unpin step {{ pinnedStep }}
              </button>
              <button v-if="anyZoomed" type="button" class="cmp-board-btn" @click="resetAllZoom">
                Reset zoom
              </button>
            </div>
            <div class="cmp-charts">
              <MetricOverlayChart
                v-for="m in chartMetrics"
                v-show="!emptyMetrics.has(m)"
                :key="m"
                :metric="m"
                :runs="selectedRefs"
                :output-dir="outputDir"
                :smoothing="smoothing"
                :log-scale="logScale"
                :refresh-token="refreshToken"
                :pinned-step="pinnedStep"
                :reset-token="resetToken"
                sync-key="compare"
                class="cmp-chart"
                @data-state="onChartDataState"
                @pin="onPin"
                @zoom-state="onZoomState"
              />
            </div>
          </div>

          <!-- Hparams -->
          <el-card class="cmp-card">
            <template #header>
              <div class="cmp-card-head">
                <span>Hyperparameters</span>
                <label class="cmp-ctrl">
                  <input v-model="onlyDiffs" type="checkbox" /> Only differing hparams
                </label>
              </div>
            </template>
            <div class="table-scroll cmp-hparams-scroll">
              <table class="cmp-table">
                <thead>
                  <tr>
                    <th class="k-col">Param</th>
                    <th v-for="r in selectedRuns" :key="r.run_id">
                      <span class="th-swatch" :style="{ background: colorMap[r.run_id] }"></span>{{ r.name }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="col in visibleHparamCols" :key="col.key" :class="{ varies: col.varies }">
                    <td class="k-col">{{ col.key }}</td>
                    <td v-for="r in selectedRuns" :key="r.run_id">{{ fmt(r.hparams[col.key]) }}</td>
                  </tr>
                  <tr v-if="!visibleHparamCols.length">
                    <td :colspan="selectedRuns.length + 1" class="muted">No differing hyperparameters</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </el-card>

          <!-- Summary metrics -->
          <el-card v-if="summaryKeys.length" class="cmp-card">
            <template #header><span>Summary metrics</span></template>
            <div class="table-scroll">
              <table class="cmp-table">
                <thead>
                  <tr>
                    <th class="k-col">Metric</th>
                    <th v-for="r in selectedRuns" :key="r.run_id">{{ r.name }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="key in summaryKeys" :key="key">
                    <td class="k-col">{{ key }}</td>
                    <td v-for="r in selectedRuns" :key="r.run_id">{{ fmt(summaryValue(r, key)) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </el-card>

          <!-- Images -->
          <el-card class="cmp-card">
            <template #header><span>Preview images</span></template>
            <RunComparePreviews :runs="selectedRefs" :output-dir="outputDir" />
          </el-card>

          <!-- Lineage + timeline -->
          <el-card class="cmp-card">
            <template #header><span>Lineage &amp; timeline</span></template>
            <div v-for="r in selectedRuns" :key="r.run_id" class="run-block">
              <h4>
                <span class="th-swatch" :style="{ background: colorMap[r.run_id] }"></span>{{ r.name }}
              </h4>
              <div class="lineage">
                <el-tag size="small" effect="plain">{{ gitField(r, "commit").slice(0, 10) || "no git" }}</el-tag>
                <el-tag v-if="gitField(r, 'branch')" size="small" effect="plain" type="info">
                  {{ gitField(r, "branch") }}
                </el-tag>
                <el-tag v-if="gitDirty(r)" size="small" type="warning" effect="plain">dirty</el-tag>
                <el-text v-if="hardwareLabel(r)" size="small" type="info">{{ hardwareLabel(r) }}</el-text>
              </div>
              <ul class="timeline">
                <li v-for="(ev, i) in timelines[r.run_id] || []" :key="i">
                  <span class="ev-type">{{ ev.type }}</span>
                  <span v-if="ev.step != null" class="ev-step">@{{ ev.step }}</span>
                  <span class="ev-ts">{{ ev.ts }}</span>
                </li>
                <li v-if="!(timelines[r.run_id] || []).length" class="muted">No events recorded</li>
              </ul>
            </div>
          </el-card>
        </template>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { Refresh } from "@element-plus/icons-vue";
import { api } from "../api";
import MetricOverlayChart from "../components/MetricOverlayChart.vue";
import RunComparePreviews from "../components/RunComparePreviews.vue";
import { buildRunColorMap } from "../lib/runColors";
import type { CompareRunRow, CompareRunsResult, TimelineEvent } from "../types/api";

const route = useRoute();
const router = useRouter();

const loading = ref(true);
const error = ref("");
const allRuns = ref<CompareRunRow[]>([]);
const metrics = ref<string[]>([]);
const timelines = ref<Record<string, TimelineEvent[]>>({});
const colorMap = ref<Record<string, string>>({});

const selectedIds = ref<string[]>([]);
const smoothing = ref(0);
const logScale = ref(false);
const onlyDiffs = ref(true);
const search = ref("");
const folderInput = ref("output");

type SortKey = "name" | "updated" | "created" | "loss" | "status";
// Default to newest-first: most recently created runs on top, older ones below.
const sortKey = ref<SortKey>("created");
const sortDir = ref<"asc" | "desc">("desc");

// TensorBoard-style refresh, mirroring RunDetailView's loss monitor: manual Reload + opt-in
// auto-update at a chosen cadence, off by default so the overlay stays put unless asked.
const autoUpdate = ref(false);
const cadenceSec = ref(10);
// Bumped on each reload so the overlay charts re-fetch their series (their selection didn't change).
const refreshToken = ref(0);
// True only during a soft (Reload / auto-update) refetch, so we dim subtly instead of swapping in
// the full-page "Loading runs…" state and losing the charts.
const refreshing = ref(false);
let autoTimer: ReturnType<typeof setInterval> | null = null;

let controller: AbortController | null = null;

// Remember the last-loaded folder so switching tabs and returning to /compare (which links here
// without a query) restores it instead of snapping back to the default "output".
const FOLDER_STORAGE_KEY = "compare.outputDir";
function rememberedFolder(): string {
  try {
    return localStorage.getItem(FOLDER_STORAGE_KEY) || "output";
  } catch {
    return "output";
  }
}
function persistFolder(dir: string): void {
  try {
    localStorage.setItem(FOLDER_STORAGE_KEY, dir);
  } catch {
    // Private mode / storage disabled — folder memory is best-effort.
  }
}

const outputDir = computed(() => {
  const q = route.query.output_dir;
  return typeof q === "string" && q ? q : rememberedFolder();
});

function parseRunsQuery(): string[] {
  const q = route.query.runs;
  return typeof q === "string" ? q.split(",").map((s) => s.trim()).filter(Boolean) : [];
}

async function loadAll(soft = false) {
  controller?.abort();
  controller = new AbortController();
  // Soft (Reload / auto-update) refetch keeps the current view; only the first/folder-change load
  // shows the blocking "Loading runs…" placeholder.
  if (soft) refreshing.value = true;
  else loading.value = true;
  error.value = "";
  try {
    const res: CompareRunsResult = await api.compareRuns([], outputDir.value, controller.signal);
    allRuns.value = res.runs || [];
    metrics.value = res.metrics || [];
    timelines.value = res.timelines || {};
    colorMap.value = buildRunColorMap(allRuns.value.map((r) => r.run_id));
    const available = new Set(allRuns.value.map((r) => r.run_id));
    selectedIds.value = parseRunsQuery().filter((id) => available.has(id));
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return;
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
    refreshing.value = false;
  }
}

// Re-fetch the folder's run rows (statuses, last scalars) AND force the overlay charts to reload
// their series — neither happens on its own because the selected run ids are unchanged.
function reload(): void {
  void loadAll(true);
  refreshToken.value++;
}

function clearAutoTimer(): void {
  if (autoTimer) clearInterval(autoTimer);
  autoTimer = null;
}
function restartAutoTimer(): void {
  clearAutoTimer();
  if (autoUpdate.value && cadenceSec.value > 0) {
    autoTimer = setInterval(reload, cadenceSec.value * 1000);
  }
}
// Enabling auto-update refreshes once immediately, then on the cadence; changing the cadence just
// reschedules without an extra fetch.
watch(autoUpdate, (on) => {
  if (on) reload();
  restartAutoTimer();
});
watch(cadenceSec, restartAutoTimer);
onUnmounted(clearAutoTimer);

function toggleSortDir(): void {
  sortDir.value = sortDir.value === "asc" ? "desc" : "asc";
}

function lossOf(r: CompareRunRow): number {
  const v = r.last_scalars?.["train/loss"];
  // Runs without a loss sort last (largest), so they don't crowd the top of an ascending list.
  return typeof v === "number" && Number.isFinite(v) ? v : Number.POSITIVE_INFINITY;
}
function compareRunsBy(a: CompareRunRow, b: CompareRunRow, key: SortKey): number {
  switch (key) {
    case "name":
      return a.name.localeCompare(b.name);
    case "status":
      return a.status.localeCompare(b.status) || a.name.localeCompare(b.name);
    case "created":
      return (a.created_at || "").localeCompare(b.created_at || "");
    case "updated":
      return (a.updated_at || "").localeCompare(b.updated_at || "");
    case "loss": {
      const av = lossOf(a);
      const bv = lossOf(b);
      return av === bv ? a.name.localeCompare(b.name) : av < bv ? -1 : 1;
    }
  }
  return 0;
}

onMounted(() => {
  folderInput.value = outputDir.value;
  persistFolder(outputDir.value);
  loadAll();
});

// Reload only when the FOLDER changes (selection lives in the URL too but must not refetch).
watch(outputDir, () => {
  folderInput.value = outputDir.value;
  persistFolder(outputDir.value);
  loadAll();
});

function applyFolder() {
  router.push({ path: "/compare", query: { output_dir: folderInput.value.trim() || "output" } });
}

function syncSelectionToUrl() {
  router.replace({
    path: "/compare",
    query: { output_dir: outputDir.value, runs: selectedIds.value.join(",") || undefined },
  });
}

function toggleRun(id: string) {
  const i = selectedIds.value.indexOf(id);
  if (i >= 0) selectedIds.value.splice(i, 1);
  else selectedIds.value.push(id);
  syncSelectionToUrl();
}
function clearSelection() {
  selectedIds.value = [];
  syncSelectionToUrl();
}
function selectAllVisible() {
  selectedIds.value = sidebarRuns.value.map((r) => r.run_id);
  syncSelectionToUrl();
}
function selectRecent(n: number) {
  const byRecent = allRuns.value
    .slice()
    .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
  selectedIds.value = byRecent.slice(0, n).map((r) => r.run_id);
  syncSelectionToUrl();
}

const sidebarRuns = computed(() => {
  const q = search.value.trim().toLowerCase();
  const filtered = q
    ? allRuns.value.filter(
        (r) => r.name.toLowerCase().includes(q) || r.run_id.toLowerCase().includes(q)
      )
    : allRuns.value.slice();
  const dir = sortDir.value === "asc" ? 1 : -1;
  filtered.sort((a, b) => dir * compareRunsBy(a, b, sortKey.value));
  return filtered;
});

const selectedRuns = computed(() =>
  allRuns.value.filter((r) => selectedIds.value.includes(r.run_id))
);
const selectedRefs = computed(() =>
  selectedRuns.value.map((r) => ({
    id: r.run_id,
    name: r.name,
    color: colorMap.value[r.run_id] || "#888888",
  }))
);

const chartMetrics = computed(() => {
  const sel = selectedRuns.value;
  if (!sel.length) return [];
  return metrics.value.filter((m) =>
    sel.some((r) => (r.tags || []).includes(m) || (r.tags || []).length === 0)
  );
});

// Metrics whose chart reported no data for the current selection — hidden so empty cards (e.g.
// val/* without an eval dataset, or grad_norm for an optimizer that doesn't expose it) don't show.
const emptyMetrics = ref<Set<string>>(new Set());
function onChartDataState(payload: { metric: string; hasData: boolean }) {
  const has = emptyMetrics.value.has(payload.metric);
  if (payload.hasData === !has) return; // no change
  const next = new Set(emptyMetrics.value);
  if (payload.hasData) next.delete(payload.metric);
  else next.add(payload.metric);
  emptyMetrics.value = next;
}
// A metric can LEAVE chartMetrics (a selected run doesn't list it) and later RE-ENTER for a run that
// does — its chart then remounts fresh. A stale "empty" flag from a previous selection would keep
// the remounted chart hidden (display:none), and a hidden chart can't lazy-load to clear its own
// flag, so it stays hidden forever. Drop the flag for any metric that enters or leaves chartMetrics
// so the fresh chart starts visible and reports its real data-state; metrics that simply stay shown
// keep their flag (no full reset, so no flicker on every selection change).
watch(chartMetrics, (next, prev) => {
  if (!emptyMetrics.value.size) return;
  const prevSet = new Set(prev);
  const nextSet = new Set(next);
  const cleaned = new Set([...emptyMetrics.value].filter((m) => nextSet.has(m) && prevSet.has(m)));
  if (cleaned.size !== emptyMetrics.value.size) emptyMetrics.value = cleaned;
});

// Global board controls. Zoom + cursor are synced across boards (uPlot sync), so zoom-reset and the
// pinned point are shared state rather than per-chart.
const pinnedStep = ref<number | null>(null);
const resetToken = ref(0);
const zoomedMetrics = ref<Set<string>>(new Set());
const anyZoomed = computed(() => zoomedMetrics.value.size > 0);

function onZoomState(payload: { metric: string; zoomed: boolean }) {
  const next = new Set(zoomedMetrics.value);
  if (payload.zoomed) next.add(payload.metric);
  else next.delete(payload.metric);
  zoomedMetrics.value = next;
}
function onPin(step: number) {
  pinnedStep.value = step;
}
function unpin() {
  pinnedStep.value = null;
}
function resetAllZoom() {
  resetToken.value += 1;
}

const hparamCols = computed(() => {
  const seen = new Map<string, Set<string>>();
  for (const r of selectedRuns.value) {
    for (const [k, v] of Object.entries(r.hparams || {})) {
      if (!seen.has(k)) seen.set(k, new Set());
      seen.get(k)!.add(JSON.stringify(v));
    }
  }
  return [...seen.entries()]
    .map(([key, vals]) => ({ key, varies: vals.size > 1 }))
    .sort((a, b) => a.key.localeCompare(b.key));
});
const visibleHparamCols = computed(() =>
  onlyDiffs.value ? hparamCols.value.filter((c) => c.varies) : hparamCols.value
);

const summaryKeys = computed(() => {
  const keys = new Set<string>();
  for (const r of selectedRuns.value) {
    for (const k of Object.keys(r.summary || {})) keys.add(k);
    for (const k of Object.keys(r.system_summary || {})) keys.add(`system/${k}`);
  }
  return Array.from(keys).sort();
});

function summaryValue(r: CompareRunRow, key: string): number | string | null {
  if (key.startsWith("system/")) return r.system_summary?.[key.slice("system/".length)] ?? null;
  return r.summary?.[key] ?? null;
}

function fmt(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(3);
    return String(Math.round(value * 1e6) / 1e6);
  }
  return String(value);
}

function gitField(r: CompareRunRow, field: string): string {
  const git = (r.lineage?.git ?? {}) as Record<string, unknown>;
  const value = git[field];
  return value == null ? "" : String(value);
}
function gitDirty(r: CompareRunRow): boolean {
  const git = (r.lineage?.git ?? {}) as Record<string, unknown>;
  return git.dirty === true;
}
function hardwareLabel(r: CompareRunRow): string {
  const hw = (r.hardware ?? {}) as Record<string, unknown>;
  const gpus = hw.gpus;
  if (Array.isArray(gpus) && gpus.length) {
    const first = gpus[0] as Record<string, unknown>;
    const name = first.name == null ? "" : String(first.name);
    const count = gpus.length > 1 ? ` ×${gpus.length}` : "";
    return name ? `${name}${count}` : "";
  }
  return "";
}
</script>

<style scoped>
.run-comparison {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
}
.page-head {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.folder-bar {
  max-width: 560px;
}
.cmp-refresh-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
.cmp-cadence {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.cmp-cadence--off {
  opacity: 0.6;
}
.cmp-cadence-input {
  width: 64px;
}

.cmp-layout {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}
.cmp-sidebar {
  flex: 0 0 260px;
  position: sticky;
  top: 8px;
  max-height: calc(100vh - 24px);
  overflow: auto;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: var(--el-bg-color);
}
.cmp-search {
  width: 100%;
  padding: 5px 8px;
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  background: var(--el-fill-color-blank);
  color: var(--el-text-color-primary);
  font-size: 13px;
}
.cmp-sidebar__sort {
  display: flex;
  gap: 6px;
}
.cmp-sort-select {
  flex: 1;
  min-width: 0;
  padding: 4px 6px;
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  background: var(--el-fill-color-blank);
  color: var(--el-text-color-regular);
  font-size: 12px;
}
.cmp-sort-dir {
  flex: 0 0 auto;
  width: 30px;
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  background: var(--el-fill-color-light);
  color: var(--el-text-color-regular);
  cursor: pointer;
  font-size: 13px;
  line-height: 1;
}
.cmp-sort-dir:hover {
  background: var(--el-fill-color);
}
.cmp-sidebar__actions {
  display: flex;
  gap: 6px;
}
.cmp-sidebar__actions button {
  flex: 1;
  font-size: 12px;
  padding: 3px 6px;
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  background: var(--el-fill-color-light);
  color: var(--el-text-color-regular);
  cursor: pointer;
}
.cmp-sidebar__actions button:hover {
  background: var(--el-fill-color);
}
.cmp-runlist {
  display: flex;
  flex-direction: column;
}
.cmp-run {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 4px;
  border-radius: 4px;
  cursor: pointer;
}
.cmp-run:hover {
  background: var(--el-fill-color-light);
}
.cmp-run__swatch {
  width: 11px;
  height: 11px;
  border-radius: 3px;
  flex: 0 0 auto;
}
.cmp-run__body {
  display: flex;
  flex-direction: column;
  min-width: 0;
  line-height: 1.25;
}
.cmp-run__name {
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cmp-run__meta {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: var(--el-text-color-secondary);
}
.cmp-run__status.is-running {
  color: var(--el-color-primary);
}
.cmp-run__status.is-failed {
  color: var(--el-color-danger);
}
.cmp-run__status.is-finished {
  color: var(--el-color-success);
}
.cmp-runlist__empty {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  padding: 8px 4px;
}

.cmp-content {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.cmp-placeholder {
  padding: 40px 0;
  text-align: center;
}
.cmp-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 18px;
  padding: 8px 12px;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
}
.cmp-ctrl {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.cmp-ctrl__val {
  font-variant-numeric: tabular-nums;
  color: var(--el-text-color-secondary);
  min-width: 2.5em;
}
.cmp-boards {
  position: relative;
}
/* Global zoom/pin controls that stay reachable while scrolling the boards: a 0-height sticky strip
   so it doesn't push the grid down, with the buttons floating over the top-right corner. */
.cmp-board-tools {
  position: sticky;
  top: 8px;
  z-index: 6;
  height: 0;
  overflow: visible;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  pointer-events: none;
}
.cmp-board-tools > * {
  pointer-events: auto;
}
.cmp-board-btn {
  font-size: 12px;
  padding: 3px 10px;
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  background: var(--el-bg-color-overlay, var(--el-fill-color-light));
  color: var(--el-text-color-regular);
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.cmp-board-btn:hover {
  background: var(--el-fill-color);
}
/* The pin is easy to forget about, so its Unpin stands out: filled accent, bolder, tabular step. */
.cmp-board-btn--unpin {
  background: var(--el-color-warning);
  border-color: var(--el-color-warning);
  color: #fff;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.cmp-board-btn--unpin:hover {
  background: var(--el-color-warning-dark-2, var(--el-color-warning));
  filter: brightness(1.05);
}
.cmp-charts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 16px;
}
.cmp-chart {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 10px;
}

.cmp-card {
  width: 100%;
}
.cmp-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.table-scroll {
  overflow-x: auto;
}
/* A wide full-hparams table buried its horizontal scrollbar at the very bottom (you had to scroll
   past every row to reach it). Cap the height so it scrolls within a box — the horizontal scrollbar
   then stays at the box's bottom edge, always reachable — and keep the header row pinned. */
.cmp-hparams-scroll {
  max-height: 60vh;
  overflow: auto;
}
.cmp-hparams-scroll .cmp-table thead th {
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--el-bg-color);
}
.cmp-hparams-scroll .cmp-table thead th.k-col {
  z-index: 3; /* above both the sticky row and the sticky first column where they cross */
}
.cmp-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}
.cmp-table th,
.cmp-table td {
  border: 1px solid var(--el-border-color-lighter);
  padding: 4px 10px;
  text-align: left;
  white-space: nowrap;
}
.cmp-table .k-col {
  font-weight: 600;
  position: sticky;
  left: 0;
  background: var(--el-bg-color);
}
/* Differing hyperparameters: a subtle on-brand cyan wash (was a muddy warning brown). */
.cmp-table tr.varies td {
  background: var(--el-color-primary-light-9);
}
.cmp-table tr.varies td.k-col {
  background: var(--el-color-primary-light-9);
  box-shadow: inset 2px 0 0 var(--el-color-primary);
}
.th-swatch {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 5px;
}
.muted {
  color: var(--el-text-color-secondary);
}
.run-block {
  padding: 8px 0;
  border-top: 1px solid var(--el-border-color-lighter);
}
.run-block h4 {
  margin: 4px 0;
}
.lineage {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.timeline {
  margin: 0;
  padding-left: 16px;
  font-size: 12px;
  color: var(--el-text-color-regular);
}
.timeline .ev-type {
  font-weight: 600;
}
.timeline .ev-step,
.timeline .ev-ts {
  margin-left: 6px;
  color: var(--el-text-color-secondary);
}
</style>

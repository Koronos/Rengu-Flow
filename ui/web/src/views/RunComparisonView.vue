<template>
  <div class="run-comparison">
    <div class="page-head">
      <h2>Compare runs</h2>
      <el-text v-if="allRuns.length" type="info" size="small">
        {{ selectedIds.length }}/{{ allRuns.length }} selected in “{{ outputDir }}”
      </el-text>
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
              <input v-model.number="smoothing" type="range" min="0" max="0.95" step="0.05" />
              <span class="cmp-ctrl__val">{{ smoothing.toFixed(2) }}</span>
            </label>
            <label class="cmp-ctrl"><input v-model="logScale" type="checkbox" /> Log scale (y)</label>
            <label class="cmp-ctrl"><input v-model="onlyDiffs" type="checkbox" /> Only differing hparams</label>
          </div>

          <!-- Overlay charts -->
          <div v-if="chartMetrics.length" class="cmp-charts">
            <MetricOverlayChart
              v-for="m in chartMetrics"
              :key="m"
              :metric="m"
              :runs="selectedRefs"
              :output-dir="outputDir"
              :smoothing="smoothing"
              :log-scale="logScale"
              sync-key="compare"
              class="cmp-chart"
            />
          </div>

          <!-- Hparams -->
          <el-card class="cmp-card">
            <template #header><span>Hyperparameters</span></template>
            <div class="table-scroll">
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
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
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

let controller: AbortController | null = null;

const outputDir = computed(() => {
  const q = route.query.output_dir;
  return typeof q === "string" && q ? q : "output";
});

function parseRunsQuery(): string[] {
  const q = route.query.runs;
  return typeof q === "string" ? q.split(",").map((s) => s.trim()).filter(Boolean) : [];
}

async function loadAll() {
  controller?.abort();
  controller = new AbortController();
  loading.value = true;
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
  }
}

onMounted(() => {
  folderInput.value = outputDir.value;
  loadAll();
});

// Reload only when the FOLDER changes (selection lives in the URL too but must not refetch).
watch(outputDir, () => {
  folderInput.value = outputDir.value;
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
  selectedIds.value = allRuns.value.map((r) => r.run_id).slice(-n);
  syncSelectionToUrl();
}

const sidebarRuns = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return allRuns.value;
  return allRuns.value.filter(
    (r) => r.name.toLowerCase().includes(q) || r.run_id.toLowerCase().includes(q)
  );
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
.table-scroll {
  overflow-x: auto;
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
.cmp-table tr.varies td {
  background: var(--el-color-warning-light-9);
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

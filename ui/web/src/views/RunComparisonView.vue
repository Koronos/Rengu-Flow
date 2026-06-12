<template>
  <div class="run-comparison">
    <div class="page-head">
      <h2>Compare runs</h2>
      <el-text v-if="runs.length" type="info" size="small">{{ runs.length }} runs</el-text>
    </div>

    <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
    <div v-else-if="loading" class="loading">
      <el-text type="info">Loading comparison…</el-text>
    </div>
    <el-empty v-else-if="!runs.length" description="No tracked runs to compare" />

    <template v-else>
      <div class="run-chips">
        <el-tag v-for="r in runs" :key="r.run_id" :type="statusType(r.status)" effect="light">
          {{ r.name }} · {{ r.status }}
        </el-tag>
      </div>

      <el-card class="cmp-card">
        <template #header>
          <div class="card-head">
            <span>Hyperparameters</span>
            <el-checkbox v-model="onlyDiffs" size="small">Only differences</el-checkbox>
          </div>
        </template>
        <div class="table-scroll">
          <table class="cmp-table">
            <thead>
              <tr>
                <th class="k-col">Param</th>
                <th v-for="r in runs" :key="r.run_id">{{ r.name }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="col in visibleHparamCols" :key="col.key" :class="{ varies: col.varies }">
                <td class="k-col">{{ col.key }}</td>
                <td v-for="r in runs" :key="r.run_id">{{ fmt(r.hparams[col.key]) }}</td>
              </tr>
              <tr v-if="!visibleHparamCols.length">
                <td :colspan="runs.length + 1" class="muted">No differing hyperparameters</td>
              </tr>
            </tbody>
          </table>
        </div>
      </el-card>

      <el-card v-if="summaryKeys.length" class="cmp-card">
        <template #header><span>Summary metrics</span></template>
        <div class="table-scroll">
          <table class="cmp-table">
            <thead>
              <tr>
                <th class="k-col">Metric</th>
                <th v-for="r in runs" :key="r.run_id">{{ r.name }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="key in summaryKeys" :key="key">
                <td class="k-col">{{ key }}</td>
                <td v-for="r in runs" :key="r.run_id">{{ fmt(summaryValue(r, key)) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </el-card>

      <el-card v-if="metrics.length" class="cmp-card">
        <template #header>
          <span>Metrics ({{ metrics.length }}) — each loads as it scrolls into view</span>
        </template>
        <LazyMetricChart
          v-for="tag in metrics"
          :key="tag"
          :tag="tag"
          :runs="runRefs"
          :output-dir="outputDir"
          class="metric-block"
        />
      </el-card>

      <el-card class="cmp-card">
        <template #header><span>Lineage &amp; timeline</span></template>
        <div v-for="r in runs" :key="r.run_id" class="run-block">
          <h4>{{ r.name }}</h4>
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
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api";
import LazyMetricChart from "../components/LazyMetricChart.vue";
import type {
  CompareColumn,
  CompareRunRow,
  CompareRunsResult,
  TimelineEvent,
} from "../types/api";

type TagType = "success" | "info" | "warning" | "danger" | "primary";

const route = useRoute();
const loading = ref(true);
const error = ref("");
const onlyDiffs = ref(true);

const runs = ref<CompareRunRow[]>([]);
const columns = ref<CompareColumn[]>([]);
const metrics = ref<string[]>([]);
const timelines = ref<Record<string, TimelineEvent[]>>({});

function selectedNames(): string[] {
  const q = route.query.runs;
  if (typeof q === "string") return q.split(",").map((s) => s.trim()).filter(Boolean);
  return [];
}

const outputDir = computed(() => {
  const q = route.query.output_dir;
  return typeof q === "string" && q ? q : "output";
});

// Stable id/name pairs handed to each lazy chart so it knows which runs to fetch.
const runRefs = computed(() => runs.value.map((r) => ({ id: r.run_id, name: r.name })));

async function load() {
  loading.value = true;
  error.value = "";
  try {
    const res: CompareRunsResult = await api.compareRuns(selectedNames(), outputDir.value);
    runs.value = res.runs || [];
    columns.value = res.columns || [];
    metrics.value = res.metrics || [];
    timelines.value = res.timelines || {};
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(load);

const visibleHparamCols = computed(() =>
  onlyDiffs.value ? columns.value.filter((c) => c.varies) : columns.value
);

const summaryKeys = computed(() => {
  const keys = new Set<string>();
  for (const r of runs.value) {
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

function statusType(status: string): TagType {
  switch (status) {
    case "finished":
      return "success";
    case "failed":
      return "danger";
    case "stopped":
      return "warning";
    case "running":
      return "primary";
    default:
      return "info";
  }
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
.run-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.cmp-card {
  width: 100%;
}
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
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
.muted {
  color: var(--el-text-color-secondary);
}
.metric-block {
  padding: 12px 0;
  border-top: 1px solid var(--el-border-color-lighter);
}
.metric-block:first-child {
  border-top: none;
}
.curve-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}
.curve-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.curve-name {
  font-size: 12px;
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
.timeline .ev-step {
  margin-left: 6px;
  color: var(--el-text-color-secondary);
}
.timeline .ev-ts {
  margin-left: 8px;
  color: var(--el-text-color-secondary);
}
</style>

<template>
  <div ref="root" class="overlay-chart">
    <div class="overlay-chart__head">
      <span class="overlay-chart__title">{{ metric }}</span>
      <span v-if="loading" class="overlay-chart__hint">loading…</span>
    </div>
    <div v-if="error" class="overlay-chart__state overlay-chart__state--error">{{ error }}</div>
    <div v-show="!error" class="overlay-chart__plot">
      <div ref="chartEl" class="overlay-chart__canvas"></div>
      <div ref="tip" class="overlay-chart__tip" style="display: none"></div>
    </div>
    <div v-if="!error && !loading && !hasData" class="overlay-chart__state">
      No data for the selected runs.
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { api } from "../api";
import { emaSmooth } from "../lib/smoothing";
import type { ScalarMetricPoint } from "../types/api";

interface RunRef {
  id: string;
  name: string;
  color: string;
}

const props = defineProps<{
  metric: string;
  runs: RunRef[];
  outputDir: string;
  smoothing: number;
  logScale: boolean;
  syncKey: string;
  /** Bumped by the parent (Reload / auto-update) to force a series re-fetch without an id change. */
  refreshToken?: number;
}>();

// Reported after each load/render so the parent can hide a metric that has no data for the
// selected runs (e.g. val/* when no eval dataset ran). Emitted both ways so a metric reappears
// once a newly selected run does have it.
const emit = defineEmits<{
  (e: "data-state", payload: { metric: string; hasData: boolean }): void;
}>();

const HEIGHT = 200;

const root = ref<HTMLElement | null>(null);
const chartEl = ref<HTMLElement | null>(null);
const tip = ref<HTMLElement | null>(null);
const loading = ref(false);
const error = ref("");
const hasData = ref(false);

let observer: IntersectionObserver | null = null;
let resizeObs: ResizeObserver | null = null;
let controller: AbortController | null = null;
let plot: uPlot | null = null;
let seriesByRun: Record<string, ScalarMetricPoint[]> = {};
let started = false;
let currentSig = "";
// Per-run wall-clock aligned to the chart's x-axis, so the hover tooltip can show elapsed time
// (point wall_time minus the run's first wall_time). Rebuilt alongside the plotted data.
let tipMeta: { names: string[]; colors: string[]; walls: (number | null)[][]; base: (number | null)[] } | null =
  null;

function cssVar(name: string, fallback: string): string {
  if (typeof getComputedStyle === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function buildData(): uPlot.AlignedData {
  const stepSet = new Set<number>();
  const smoothed: Record<string, ScalarMetricPoint[]> = {};
  for (const r of props.runs) {
    const raw = seriesByRun[r.id] || [];
    const s = props.smoothing > 0 ? emaSmooth(raw, props.smoothing) : raw;
    smoothed[r.id] = s;
    for (const p of s) stepSet.add(p.step);
  }
  const xs = [...stepSet].sort((a, b) => a - b);
  const xIndex = new Map(xs.map((step, i) => [step, i]));
  const ys: (number | null)[][] = props.runs.map((r) => {
    const arr: (number | null)[] = new Array(xs.length).fill(null);
    for (const p of smoothed[r.id]) {
      let v: number | null = p.value;
      if (!Number.isFinite(v as number)) v = null;
      if (props.logScale && v != null && !(v > 0)) v = null; // log can't show <= 0
      const i = xIndex.get(p.step);
      if (i != null) arr[i] = v;
    }
    return arr;
  });
  const walls: (number | null)[][] = props.runs.map((r) => {
    const arr: (number | null)[] = new Array(xs.length).fill(null);
    for (const p of smoothed[r.id]) {
      const i = xIndex.get(p.step);
      if (i != null) arr[i] = typeof p.wall_time === "number" ? p.wall_time : null;
    }
    return arr;
  });
  const base = walls.map((w) => w.find((v) => v != null) ?? null);
  tipMeta = {
    names: props.runs.map((r) => r.name),
    colors: props.runs.map((r) => r.color),
    walls,
    base,
  };
  return [xs, ...ys] as uPlot.AlignedData;
}

function fmtVal(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return v.toExponential(3);
  return String(Math.round(v * 1e6) / 1e6);
}

function fmtDuration(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${ss}s`;
  return `${ss}s`;
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string
  );
}

// TensorBoard-style hover popup: step, each run's (smoothed) value, and elapsed wall time.
function updateTip(u: uPlot): void {
  const el = tip.value;
  if (!el) return;
  const idx = u.cursor.idx;
  if (idx == null || tipMeta == null) {
    el.style.display = "none";
    return;
  }
  const step = (u.data[0] as number[])[idx];
  let rows = "";
  for (let s = 1; s < u.data.length; s++) {
    const val = (u.data[s] as (number | null)[])[idx];
    const wall = tipMeta.walls[s - 1]?.[idx] ?? null;
    const base = tipMeta.base[s - 1];
    const rel = wall != null && base != null ? wall - base : null;
    rows +=
      `<div class="tip-row"><span class="tip-sw" style="background:${tipMeta.colors[s - 1]}"></span>` +
      `<span class="tip-name">${escapeHtml(tipMeta.names[s - 1])}</span>` +
      `<span class="tip-val">${fmtVal(val)}</span>` +
      `<span class="tip-dt">${fmtDuration(rel)}</span></div>`;
  }
  el.innerHTML = `<div class="tip-head">step ${step}</div>${rows}`;
  el.style.display = "block";
  // Flip to the cursor's left near the right edge so the popup stays inside the plot.
  const plotW = u.over?.clientWidth ?? u.width ?? 600;
  const left = u.cursor.left ?? 0;
  const top = u.cursor.top ?? 0;
  const flip = left > plotW / 2;
  el.style.left = flip ? "auto" : `${left + 14}px`;
  el.style.right = flip ? `${plotW - left + 14}px` : "auto";
  el.style.top = `${Math.max(0, top + 12)}px`;
}

function makeOpts(width: number): uPlot.Options {
  const axisColor = cssVar("--el-text-color-secondary", "#909399");
  const gridColor = cssVar("--el-border-color-lighter", "#ebeef5");
  const axis = {
    stroke: axisColor,
    grid: { stroke: gridColor, width: 1 },
    ticks: { stroke: gridColor, width: 1 },
  };
  return {
    width,
    height: HEIGHT,
    scales: { x: { time: false }, y: { distr: props.logScale ? 3 : 1 } },
    axes: [axis, axis],
    cursor: { sync: { key: props.syncKey }, points: { size: 5 } },
    legend: { show: true },
    hooks: {
      setCursor: [updateTip],
      setData: [updateTip],
    },
    series: [
      { label: "step" },
      ...props.runs.map((r) => ({
        label: r.name,
        stroke: r.color,
        width: 1.5,
        points: { show: false },
        // Each run contributes its own steps to the shared x-axis, so every other run's steps land
        // as nulls in this series. spanGaps must bridge them, else a run whose step grid interleaves
        // with another's renders as disconnected (invisible) points instead of a continuous line.
        spanGaps: true,
      })),
    ],
  };
}

function sig(): string {
  return props.runs.map((r) => `${r.id}:${r.color}`).join(",") + `|${props.logScale}`;
}

function destroyPlot() {
  plot?.destroy();
  plot = null;
}

function render() {
  if (!chartEl.value) return;
  const data = buildData();
  hasData.value = (data[0] as number[]).length > 0;
  emit("data-state", { metric: props.metric, hasData: hasData.value });
  if (!hasData.value) {
    destroyPlot();
    return;
  }
  const nextSig = sig();
  if (plot && nextSig === currentSig) {
    plot.setData(data); // fast path (e.g. smoothing slider)
    return;
  }
  destroyPlot();
  currentSig = nextSig;
  plot = new uPlot(makeOpts(chartEl.value.clientWidth || 600), data, chartEl.value);
}

async function loadSeries() {
  controller?.abort();
  const ids = props.runs.map((r) => r.id);
  if (!ids.length) {
    seriesByRun = {};
    render();
    return;
  }
  controller = new AbortController();
  loading.value = true;
  error.value = "";
  try {
    const res = await api.runSeries(ids, props.metric, 600, props.outputDir, controller.signal);
    seriesByRun = res.series || {};
    render();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return; // expected on unmount/change
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  if (!root.value) return;
  const begin = () => {
    if (started) return;
    started = true;
    loadSeries();
  };
  if (typeof IntersectionObserver === "undefined") {
    begin();
  } else {
    observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          begin();
          observer?.disconnect();
          observer = null;
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(root.value);
  }
  if (typeof ResizeObserver !== "undefined" && chartEl.value) {
    resizeObs = new ResizeObserver(() => {
      if (plot && chartEl.value) plot.setSize({ width: chartEl.value.clientWidth || 600, height: HEIGHT });
    });
    resizeObs.observe(chartEl.value);
  }
});

// Selection change → re-fetch (aborting the previous load). Only if we've already started loading.
watch(
  () => props.runs.map((r) => r.id).join(","),
  () => {
    if (started) loadSeries();
  }
);
// Smoothing / log-scale change → re-render from cached series (no fetch).
watch(
  () => [props.smoothing, props.logScale],
  () => {
    if (started) render();
  }
);
// Parent asked for a refresh (Reload button / auto-update tick) → re-fetch the series.
watch(
  () => props.refreshToken,
  () => {
    if (started) loadSeries();
  }
);

onBeforeUnmount(() => {
  controller?.abort();
  observer?.disconnect();
  resizeObs?.disconnect();
  destroyPlot();
});
</script>

<style scoped>
.overlay-chart {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 240px;
}
.overlay-chart__head {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.overlay-chart__title {
  font-weight: 600;
  font-size: 13px;
}
.overlay-chart__hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.overlay-chart__plot {
  position: relative;
}
.overlay-chart__canvas {
  width: 100%;
}
.overlay-chart__tip {
  position: absolute;
  z-index: 10;
  pointer-events: none;
  min-width: 160px;
  padding: 6px 8px;
  border: 1px solid var(--el-border-color);
  border-radius: 5px;
  background: var(--el-bg-color-overlay, var(--el-bg-color));
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25);
  font-size: 12px;
  line-height: 1.5;
}
.overlay-chart__tip :deep(.tip-head) {
  font-weight: 600;
  margin-bottom: 3px;
  color: var(--el-text-color-primary);
}
.overlay-chart__tip :deep(.tip-row) {
  display: grid;
  grid-template-columns: 10px 1fr auto auto;
  align-items: center;
  gap: 6px;
  color: var(--el-text-color-regular);
}
.overlay-chart__tip :deep(.tip-sw) {
  width: 10px;
  height: 10px;
  border-radius: 2px;
}
.overlay-chart__tip :deep(.tip-name) {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 160px;
}
.overlay-chart__tip :deep(.tip-val) {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
}
.overlay-chart__tip :deep(.tip-dt) {
  font-variant-numeric: tabular-nums;
  color: var(--el-text-color-secondary);
}
.overlay-chart__state {
  color: var(--el-text-color-secondary);
  font-size: 13px;
  padding: 8px 0;
}
.overlay-chart__state--error {
  color: var(--el-color-danger);
}
/* uPlot legend tweaks so many-run legends stay compact. */
.overlay-chart :deep(.u-legend) {
  font-size: 12px;
}
.overlay-chart :deep(.u-legend .u-marker) {
  width: 10px;
  height: 10px;
}
</style>

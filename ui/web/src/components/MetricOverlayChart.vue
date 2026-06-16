<template>
  <div ref="root" class="overlay-chart">
    <div class="overlay-chart__head">
      <span class="overlay-chart__title">{{ metric }}</span>
      <span v-if="loading" class="overlay-chart__hint">loading…</span>
    </div>
    <div v-if="error" class="overlay-chart__state overlay-chart__state--error">{{ error }}</div>
    <div v-show="!error" ref="chartEl" class="overlay-chart__canvas"></div>
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

const HEIGHT = 200;

const root = ref<HTMLElement | null>(null);
const chartEl = ref<HTMLElement | null>(null);
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
  return [xs, ...ys] as uPlot.AlignedData;
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
    series: [
      { label: "step" },
      ...props.runs.map((r) => ({
        label: r.name,
        stroke: r.color,
        width: 1.5,
        points: { show: false },
        spanGaps: false,
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
.overlay-chart__canvas {
  width: 100%;
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

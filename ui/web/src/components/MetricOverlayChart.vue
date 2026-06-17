<template>
  <div ref="root" class="overlay-chart">
    <div class="overlay-chart__head">
      <span class="overlay-chart__title">{{ metric }}</span>
      <span v-if="loading" class="overlay-chart__hint">loading…</span>
      <!-- Which colour is which run — the always-visible "signal" for the readout values below. -->
      <div class="overlay-chart__legend">
        <span v-for="r in runs" :key="r.id" class="ov-leg" :title="r.name">
          <span class="ov-leg__sw" :style="{ background: r.color }"></span>
          <span class="ov-leg__name">{{ r.name }}</span>
        </span>
      </div>
    </div>
    <div v-if="error" class="overlay-chart__state overlay-chart__state--error">{{ error }}</div>
    <div v-show="!error" class="overlay-chart__plot">
      <div ref="chartEl" class="overlay-chart__canvas"></div>
    </div>
    <!-- Hover readout BELOW the plot (not over it) so it never covers the curve being inspected. -->
    <div ref="tip" class="overlay-chart__tip"></div>
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
  /** Pinned x (step) shared across all boards: when set, the readout freezes here regardless of
   *  hover, so you can scroll away without losing the inspected point. null = follow the cursor. */
  pinnedStep?: number | null;
  /** Bumped by the parent's global "Reset zoom" to restore the full x range on every board. */
  resetToken?: number;
}>();

const emit = defineEmits<{
  // Reported after each load/render so the parent can hide a metric that has no data for the
  // selected runs (e.g. val/* when no eval dataset ran).
  (e: "data-state", payload: { metric: string; hasData: boolean }): void;
  // Clicking a board pins this x (step) across all boards (cursor is synced).
  (e: "pin", step: number): void;
  // Whether THIS board is zoomed in, so the parent can show one global "Reset zoom".
  (e: "zoom-state", payload: { metric: string; zoomed: boolean }): void;
}>();

const HEIGHT = 200;

const root = ref<HTMLElement | null>(null);
const chartEl = ref<HTMLElement | null>(null);
const tip = ref<HTMLElement | null>(null);
const loading = ref(false);
const error = ref("");
const hasData = ref(false);
// True while the x-axis is zoomed in (drag-selected a sub-range), so the "Reset zoom" button shows.
const zoomed = ref(false);

let observer: IntersectionObserver | null = null;
let resizeObs: ResizeObserver | null = null;
let controller: AbortController | null = null;
let plot: uPlot | null = null;
let seriesByRun: Record<string, ScalarMetricPoint[]> = {};
let started = false;
let currentSig = "";
// Full x extent of the current data ([first step, last step]); lets the setScale hook tell whether
// the view is zoomed and lets resetZoom restore the whole range.
let xDomain: [number, number] | null = null;
// True only while the pointer is physically over THIS chart. The cursor is synced across all
// overlay charts (shared vertical line for comparison), so setCursor fires on every chart; without
// this guard each one would pop its own tooltip and we'd see N tooltips at once. The popup is shown
// only on the chart under the pointer.
let isHovered = false;
// Per-run smoothed points (sorted by step) plus first wall-clock, so the tooltip can look up each
// run's value at the nearest step to the cursor — runs rarely share exact steps, so reading the
// x-aligned cell would show a value for just one run and "—" for the rest. Rebuilt with the data.
let tipMeta:
  | { names: string[]; colors: string[]; points: ScalarMetricPoint[][]; base: (number | null)[] }
  | null = null;

/** Nearest point to ``step`` in a step-ascending series (binary search), or null if empty. */
function nearestPoint(points: ScalarMetricPoint[], step: number): ScalarMetricPoint | null {
  const n = points.length;
  if (!n) return null;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].step < step) lo = mid + 1;
    else hi = mid;
  }
  // lo is the first point with step >= target; compare it with its predecessor and keep the closer.
  if (lo > 0 && Math.abs(points[lo - 1].step - step) <= Math.abs(points[lo].step - step)) {
    return points[lo - 1];
  }
  return points[lo];
}

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
  xDomain = xs.length ? [xs[0], xs[xs.length - 1]] : null;
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
  const points = props.runs.map((r) => smoothed[r.id]);
  const base = points.map((pts) => {
    const first = pts.find((p) => typeof p.wall_time === "number");
    return first ? (first.wall_time as number) : null;
  });
  tipMeta = {
    names: props.runs.map((r) => r.name),
    colors: props.runs.map((r) => r.color),
    points,
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

/** Index in the (ascending) x array nearest to ``step``. */
function nearestIndex(xs: number[], step: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < step) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(xs[lo - 1] - step) <= Math.abs(xs[lo] - step)) return lo - 1;
  return lo;
}

// Which x to read the readout at: the pinned step (frozen, survives hover/scroll) takes precedence;
// otherwise the live cursor while the pointer is over THIS chart; otherwise nothing.
function readoutIndex(): number | null {
  if (!plot) return null;
  const xs = plot.data[0] as number[];
  if (!xs.length) return null;
  if (props.pinnedStep != null) return nearestIndex(xs, props.pinnedStep);
  if (isHovered) return plot.cursor.idx ?? null;
  return null;
}

// Readout below the plot: step + each run's (smoothed) value at the nearest step + elapsed wall
// time. Rendered for the hovered chart, or frozen on the pinned step across all boards.
function renderReadout(): void {
  const el = tip.value;
  if (!el || !plot || !tipMeta) {
    if (el) el.innerHTML = "";
    return;
  }
  const idx = readoutIndex();
  if (idx == null) {
    el.innerHTML = "";
    return;
  }
  const step = (plot.data[0] as number[])[idx];
  let rows = "";
  for (let i = 0; i < tipMeta.points.length; i++) {
    // Each run keeps its own step grid, so read the nearest point rather than the x-aligned cell
    // (which is null for every run that lacks a sample at exactly this step).
    const p = nearestPoint(tipMeta.points[i], step);
    const val = p ? p.value : null;
    const wall = p && typeof p.wall_time === "number" ? p.wall_time : null;
    const base = tipMeta.base[i];
    const rel = wall != null && base != null ? wall - base : null;
    rows +=
      `<div class="tip-row"><span class="tip-sw" style="background:${tipMeta.colors[i]}"></span>` +
      `<span class="tip-name">${escapeHtml(tipMeta.names[i])}</span>` +
      `<span class="tip-val">${fmtVal(val)}</span>` +
      `<span class="tip-dt">${fmtDuration(rel)}</span></div>`;
  }
  const pinned = props.pinnedStep != null ? ' · <span class="tip-pin">pinned</span>' : "";
  el.innerHTML = `<div class="tip-head">step ${step}${pinned}</div>${rows}`;
}

// Drag-zoom changed the x range → tell the parent whether this board is zoomed (it shows one
// global Reset zoom, since uPlot syncs the zoom across all boards).
function setZoomed(z: boolean): void {
  if (z === zoomed.value) return;
  zoomed.value = z;
  emit("zoom-state", { metric: props.metric, zoomed: z });
}

function onSetScale(u: uPlot, key: string): void {
  if (key !== "x" || !xDomain) return;
  const sc = u.scales.x;
  const eps = (xDomain[1] - xDomain[0]) * 1e-6;
  setZoomed(
    sc.min != null && sc.max != null && (sc.min > xDomain[0] + eps || sc.max < xDomain[1] - eps)
  );
}

// Restore the full x range. Rebuilding the plot (clear the sig so render recreates it) resets both
// axes to auto — simpler and more reliable than nudging scales by hand.
function resetZoom(): void {
  currentSig = "";
  render();
  setZoomed(false);
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
    // uPlot's own legend is off: it duplicated the readout below and showed a redundant inline
    // "step". The head colour legend is the signal; the strip below shows the per-step values.
    cursor: { sync: { key: props.syncKey }, points: { size: 5 } },
    legend: { show: false },
    hooks: {
      ready: [
        (u: uPlot) => {
          // Track real pointer presence so the readout shows only for the hovered chart (the synced
          // cursor fires setCursor on every board).
          u.over.addEventListener("mouseenter", () => {
            isHovered = true;
          });
          u.over.addEventListener("mouseleave", () => {
            isHovered = false;
            renderReadout(); // clears unless a step is pinned
          });
          // Click (not a drag-zoom) pins the cursor's step across all boards.
          let down = false;
          let dragged = false;
          let downX = 0;
          let downY = 0;
          u.over.addEventListener("mousedown", (e: MouseEvent) => {
            down = true;
            dragged = false;
            downX = e.clientX;
            downY = e.clientY;
          });
          u.over.addEventListener("mousemove", (e: MouseEvent) => {
            if (down && (Math.abs(e.clientX - downX) > 4 || Math.abs(e.clientY - downY) > 4)) {
              dragged = true;
            }
          });
          u.over.addEventListener("mouseup", () => {
            down = false;
          });
          u.over.addEventListener("click", () => {
            if (dragged) return; // that gesture was a drag-zoom, not a pin
            const idx = u.cursor.idx;
            if (idx != null) emit("pin", (u.data[0] as number[])[idx]);
          });
        },
      ],
      setCursor: [() => renderReadout()],
      setData: [() => renderReadout()],
      setScale: [onSetScale],
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
// Parent's global "Reset zoom" → restore the full range on this board too.
watch(
  () => props.resetToken,
  () => {
    if (plot && zoomed.value) resetZoom();
  }
);
// Pinned step changed (set/cleared by the parent) → re-render the frozen/live readout.
watch(
  () => props.pinnedStep,
  () => renderReadout()
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
  align-items: center;
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
.overlay-chart__legend {
  margin-left: auto;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  justify-content: flex-end;
  max-width: 70%;
}
.ov-leg {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--el-text-color-secondary);
  min-width: 0;
}
.ov-leg__sw {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex: 0 0 auto;
}
.ov-leg__name {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 140px;
}
.overlay-chart__plot {
  position: relative;
}
.overlay-chart__canvas {
  width: 100%;
}
/* Make uPlot's drag-to-zoom selection rectangle clearly visible (its default is barely tinted,
   which vanishes on a dark theme). */
.overlay-chart__plot :deep(.u-select) {
  background: color-mix(in srgb, var(--el-color-primary) 22%, transparent);
  border: 1px solid var(--el-color-primary);
}
/* Hover readout below the plot — a reserved strip so showing it never shifts the layout, scrolls
   when many runs are overlaid. */
.overlay-chart__tip {
  min-height: 1.6em;
  max-height: 108px;
  overflow-y: auto;
  font-size: 12px;
  line-height: 1.5;
  color: var(--el-text-color-regular);
}
.overlay-chart__tip :deep(.tip-head) {
  font-weight: 600;
  margin-bottom: 3px;
  color: var(--el-text-color-primary);
}
.overlay-chart__tip :deep(.tip-pin) {
  font-weight: 600;
  color: var(--el-color-primary);
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
</style>

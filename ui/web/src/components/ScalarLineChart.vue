<template>
  <div v-if="!points.length" class="scalar-chart-empty">
    <el-text type="info" size="small">No {{ tag || "metric" }} data yet</el-text>
  </div>
  <div v-else class="scalar-chart">
    <div class="scalar-chart-head">
      <el-text type="info" size="small">{{ tag }}</el-text>
      <el-text v-if="latestLabel" size="small" class="scalar-latest">{{ latestLabel }}</el-text>
    </div>
    <div
      ref="chartWrap"
      class="scalar-chart-wrap"
      @mousemove="onPointerMove"
      @mouseleave="onPointerLeave"
    >
      <svg
        :viewBox="`0 0 ${width} ${height}`"
        class="scalar-chart-svg"
        preserveAspectRatio="none"
      >
        <line
          v-if="hoverX != null"
          :x1="hoverX"
          :y1="pad"
          :x2="hoverX"
          :y2="height - pad"
          class="scalar-crosshair"
          vector-effect="non-scaling-stroke"
        />
        <polyline
          v-if="polylinePoints"
          :points="polylinePoints"
          fill="none"
          stroke="var(--el-color-primary)"
          stroke-width="1.5"
          vector-effect="non-scaling-stroke"
        />
        <circle
          v-if="hoverDot"
          :cx="hoverDot.x"
          :cy="hoverDot.y"
          r="3"
          class="scalar-hover-dot"
          vector-effect="non-scaling-stroke"
        />
      </svg>
      <div
        v-if="tooltipLines.length"
        class="scalar-tooltip"
        :style="tooltipStyle"
        role="status"
      >
        <div v-for="(line, i) in tooltipLines" :key="i">{{ line }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import type { PropType } from "vue";
import {
  formatScalarValue,
  formatWallTime,
  nearestPointIndex,
  type ScalarPoint,
} from "../lib/scalarChart";

const props = defineProps({
  scalars: { type: Object as PropType<Record<string, ScalarPoint[]>>, default: () => ({}) },
  tag: { type: String, default: "train/loss" },
  maxPoints: { type: Number, default: 200 },
  width: { type: Number, default: 400 },
  height: { type: Number, default: 100 },
  xAxisLabel: { type: String, default: "step" },
  valueLabel: { type: String, default: "value" },
});

const pad = 4;
const chartWrap = ref<HTMLElement | null>(null);
const hoverIndex = ref<number | null>(null);

const points = computed<ScalarPoint[]>(() => {
  const series = props.scalars?.[props.tag] || [];
  if (!series.length) return [];
  return series.slice(-props.maxPoints);
});

const valueRange = computed(() => {
  const pts = points.value;
  if (!pts.length) return { min: 0, max: 1 };
  const values = pts.map((p) => Number(p.value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max: max === min ? min + 1 : max };
});

function pointToSvg(i: number, value: number): { x: number; y: number } {
  const pts = points.value;
  const w = props.width - pad * 2;
  const h = props.height - pad * 2;
  const { min, max } = valueRange.value;
  const span = max - min || 1;
  const x =
    pts.length < 2 ? pad + w / 2 : pad + (i / (pts.length - 1)) * w;
  const y = pad + h - ((Number(value) - min) / span) * h;
  return { x, y };
}

const polylinePoints = computed(() => {
  const pts = points.value;
  if (pts.length < 2) return "";
  return pts
    .map((p, i) => {
      const { x, y } = pointToSvg(i, Number(p.value));
      return `${x},${y}`;
    })
    .join(" ");
});

const hoverDot = computed(() => {
  const idx = hoverIndex.value;
  const pts = points.value;
  if (idx == null || idx < 0 || idx >= pts.length) return null;
  return pointToSvg(idx, Number(pts[idx].value));
});

const hoverX = computed(() => hoverDot.value?.x ?? null);

const tooltipLines = computed(() => {
  const idx = hoverIndex.value;
  const pts = points.value;
  if (idx == null || idx < 0 || idx >= pts.length) return [];
  const p = pts[idx];
  const lines = [
    `${props.valueLabel}: ${formatScalarValue(Number(p.value))}`,
    `${props.xAxisLabel}: ${p.step}`,
  ];
  const ts = formatWallTime(p.wall_time);
  if (ts) lines.push(`time: ${ts}`);
  return lines;
});

const tooltipStyle = computed(() => {
  const wrap = chartWrap.value;
  const idx = hoverIndex.value;
  const pts = points.value;
  if (!wrap || idx == null || !pts.length) return {};
  const rect = wrap.getBoundingClientRect();
  const ratio = pts.length < 2 ? 0.5 : idx / (pts.length - 1);
  const leftPx = ratio * rect.width;
  const flip = leftPx > rect.width * 0.55;
  return {
    left: `${leftPx}px`,
    top: "8px",
    transform: flip ? "translate(-100%, 0)" : "translate(8px, 0)",
  };
});

const latestLabel = computed(() => {
  const pts = points.value;
  if (!pts.length) return "";
  const last = pts[pts.length - 1];
  return `${props.xAxisLabel} ${last.step}: ${formatScalarValue(Number(last.value))}`;
});

function onPointerMove(ev: MouseEvent) {
  const wrap = chartWrap.value;
  const pts = points.value;
  if (!wrap || !pts.length) return;
  const rect = wrap.getBoundingClientRect();
  hoverIndex.value = nearestPointIndex(ev.clientX, rect, pts.length);
}

function onPointerLeave() {
  hoverIndex.value = null;
}
</script>

<style scoped>
.scalar-chart-empty {
  padding: 12px 0;
}
.scalar-chart-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 4px;
}
.scalar-latest {
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  font-size: 11px;
}
.scalar-chart-wrap {
  position: relative;
  width: 100%;
  height: 100px;
  cursor: crosshair;
}
.scalar-chart-svg {
  width: 100%;
  height: 100%;
  display: block;
  background: var(--el-fill-color-lighter);
  border-radius: 4px;
}
.scalar-crosshair {
  stroke: var(--el-text-color-secondary);
  stroke-width: 1;
  stroke-dasharray: 4 3;
  opacity: 0.75;
}
.scalar-hover-dot {
  fill: var(--el-color-primary);
  stroke: var(--el-bg-color);
  stroke-width: 1.5;
}
.scalar-tooltip {
  position: absolute;
  z-index: 2;
  pointer-events: none;
  padding: 6px 8px;
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  font-size: 11px;
  line-height: 1.45;
  color: var(--el-text-color-primary);
  background: var(--el-bg-color-overlay);
  border: 1px solid var(--el-border-color);
  border-radius: 4px;
  box-shadow: var(--el-box-shadow-light);
  white-space: nowrap;
}
</style>

<template>
  <div v-if="!points.length" class="scalar-chart-empty">
    <el-text type="info" size="small">No {{ tag || "metric" }} data yet</el-text>
  </div>
  <div v-else class="scalar-chart">
    <div class="scalar-chart-head">
      <el-text type="info" size="small">{{ tag }}</el-text>
      <el-text v-if="latestLabel" size="small" class="scalar-latest">{{ latestLabel }}</el-text>
    </div>
    <svg
      :viewBox="`0 0 ${width} ${height}`"
      class="scalar-chart-svg"
      preserveAspectRatio="none"
    >
      <polyline
        v-if="polylinePoints"
        :points="polylinePoints"
        fill="none"
        stroke="var(--el-color-primary)"
        stroke-width="1.5"
        vector-effect="non-scaling-stroke"
      />
    </svg>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  scalars: { type: Object, default: () => ({}) },
  tag: { type: String, default: "train/loss" },
  maxPoints: { type: Number, default: 120 },
  width: { type: Number, default: 400 },
  height: { type: Number, default: 100 },
});

const points = computed(() => {
  const series = props.scalars?.[props.tag] || [];
  if (!series.length) return [];
  return series.slice(-props.maxPoints);
});

const polylinePoints = computed(() => {
  const pts = points.value;
  if (pts.length < 2) return "";
  const values = pts.map((p) => Number(p.value));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = 4;
  const w = props.width - pad * 2;
  const h = props.height - pad * 2;
  return pts
    .map((p, i) => {
      const x = pad + (i / (pts.length - 1)) * w;
      const y = pad + h - ((Number(p.value) - min) / span) * h;
      return `${x},${y}`;
    })
    .join(" ");
});

const latestLabel = computed(() => {
  const pts = points.value;
  if (!pts.length) return "";
  const last = pts[pts.length - 1];
  return `step ${last.step}: ${Number(last.value).toFixed(6)}`;
});
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
  font-family: ui-monospace, monospace;
  font-size: 11px;
}
.scalar-chart-svg {
  width: 100%;
  height: 100px;
  display: block;
  background: var(--el-fill-color-lighter);
  border-radius: 4px;
}
</style>

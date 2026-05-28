<template>
  <div v-if="!bars.length">
    <el-empty description="No train/loss metrics yet" :image-size="60" />
  </div>
  <div v-else class="chart-wrap">
    <el-text type="info" size="small">train/loss (last {{ bars.length }} points)</el-text>
    <div class="chart-bars">
      <el-tooltip
        v-for="(bar, i) in bars"
        :key="i"
        :content="`step ${bar.step}: ${bar.value.toFixed(6)}`"
        placement="top"
      >
        <div class="bar" :style="{ height: bar.height }" />
      </el-tooltip>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PropType } from "vue";

interface ScalarPoint {
  step: number;
  value: number;
}

const props = defineProps({
  scalars: { type: Object as PropType<Record<string, ScalarPoint[]>>, default: () => ({}) },
});

const bars = computed(() => {
  const series = props.scalars?.["train/loss"] || [];
  if (!series.length) return [];
  const max = Math.max(...series.map((p) => p.value), 1e-6);
  return series.slice(-40).map((p) => ({
    step: p.step,
    value: p.value,
    height: `${Math.max(4, (p.value / max) * 100)}%`,
  }));
});
</script>

<style scoped>
.chart-wrap {
  width: 100%;
}
.chart-bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 120px;
  margin-top: 8px;
  padding: 4px 0;
}
.bar {
  flex: 1;
  min-width: 4px;
  max-width: 12px;
  background: var(--el-color-primary);
  border-radius: 2px 2px 0 0;
  opacity: 0.85;
}
</style>

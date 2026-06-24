<template>
  <span class="steps-readout" :class="{ 'is-loading': loading }">
    <template v-if="result">
      ~{{ fmt(result.total_steps) }} steps
      <span class="steps-readout__sub">({{ fmt(result.steps_per_epoch) }}/epoch)</span>
    </template>
    <template v-else>—</template>
  </span>
</template>

<script setup lang="ts">
import type { EstimateStepsResult } from "../types/api";

defineProps<{
  result: EstimateStepsResult | null;
  loading?: boolean;
}>();

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}
</script>

<style scoped>
.steps-readout {
  font-size: 13px;
  font-family: ui-monospace, monospace;
  color: var(--el-text-color-regular);
  white-space: nowrap;
}
.steps-readout.is-loading {
  opacity: 0.5;
}
.steps-readout__sub {
  color: var(--el-text-color-secondary);
}
</style>

<template>
  <div
    class="auto-refresh-bar"
    :class="{
      'auto-refresh-bar--paused': paused,
      'auto-refresh-bar--polling': polling,
    }"
  >
    <el-button
      size="small"
      :icon="Refresh"
      :loading="refreshing"
      @click="emit('refresh')"
    />
    <el-select
      :model-value="intervalSec"
      size="small"
      class="interval-select"
      @update:model-value="onIntervalChange"
    >
      <el-option
        v-for="opt in AUTO_REFRESH_OPTIONS"
        :key="opt.value"
        :label="opt.label"
        :value="opt.value"
      />
    </el-select>
    <el-text v-if="polling && !refreshing" type="info" size="small" class="polling-hint">
      Updating…
    </el-text>
    <el-text v-else-if="lastLabel" type="info" size="small" class="last-updated">
      {{ lastLabel }}
    </el-text>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { Refresh } from "@element-plus/icons-vue";
import {
  AUTO_REFRESH_OPTIONS,
  formatLastUpdated,
  parseStoredInterval,
  type AutoRefreshIntervalSec,
} from "../lib/autoRefresh";

const props = defineProps({
  intervalSec: { type: Number, required: true },
  refreshing: { type: Boolean, default: false },
  polling: { type: Boolean, default: false },
  lastUpdated: { type: Object as () => Date | null, default: null },
  paused: { type: Boolean, default: false },
});

const emit = defineEmits<{
  "update:intervalSec": [value: AutoRefreshIntervalSec];
  refresh: [];
}>();

const lastLabel = computed(() => {
  if (props.paused) return "Paused (tab hidden)";
  if (!props.lastUpdated) return "";
  return `Updated ${formatLastUpdated(props.lastUpdated)}`;
});

function onIntervalChange(v: unknown) {
  const sec = parseStoredInterval(v == null ? null : String(v));
  emit("update:intervalSec", sec);
}
</script>

<style scoped>
.auto-refresh-bar {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.interval-select {
  width: 84px;
}
.last-updated,
.polling-hint {
  white-space: nowrap;
}
.polling-hint {
  opacity: 0.85;
}
.auto-refresh-bar--paused .last-updated {
  opacity: 0.85;
}
</style>

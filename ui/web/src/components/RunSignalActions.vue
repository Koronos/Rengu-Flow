<template>
  <div v-if="available" class="run-signal-actions">
    <el-divider v-if="showDivider" content-position="left">Signals</el-divider>
    <el-text v-if="compact" type="info" size="small" class="signal-intro">
      Control the active training run
    </el-text>
    <div v-for="group in groups" :key="group.label" class="signal-group">
      <el-text type="info" size="small" class="signal-group-label">{{ group.label }}</el-text>
      <div class="signal-grid">
        <el-tooltip
          v-for="item in group.items"
          :key="item.id"
          :content="item.hint || item.label"
          placement="top"
        >
          <el-button
            size="small"
            :type="item.variant || 'default'"
            :plain="item.variant === 'danger'"
            :disabled="disabled"
            @click="emit('send', item.id)"
          >
            {{ item.label }}
          </el-button>
        </el-tooltip>
      </div>
    </div>
  </div>
  <el-text v-else-if="showUnavailableHint" type="info" size="small" class="signals-unavailable">
    Signals are available only while training is running or stopping.
  </el-text>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useTrainingSignalDefinitions } from "../composables/useTrainingSignalDefinitions";
import { groupSignalDefinitions } from "../lib/trainingSignals";

const props = defineProps({
  available: { type: Boolean, default: false },
  diskExportWait: { type: Boolean, default: false },
  showUnavailableHint: { type: Boolean, default: true },
  showDivider: { type: Boolean, default: true },
  compact: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits<{
  send: [type: string];
}>();

const { definitions } = useTrainingSignalDefinitions();

const groups = computed(() =>
  groupSignalDefinitions(definitions.value, props.diskExportWait)
);
</script>

<style scoped>
.run-signal-actions {
  margin-top: 4px;
}
.signal-intro {
  display: block;
  margin-bottom: 8px;
}
.signal-group {
  margin-bottom: 10px;
}
.signal-group-label {
  display: block;
  margin-bottom: 6px;
}
.signal-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.signals-unavailable {
  display: block;
  margin-top: 8px;
}
</style>

<template>
  <div
    class="dataset-thumb-empty"
    :class="{
      'dataset-thumb-empty--compact': compact,
      'dataset-thumb-empty--icon-only': iconOnly,
    }"
    :title="label"
  >
    <el-icon class="dataset-thumb-empty-icon"><Picture /></el-icon>
    <span v-if="!iconOnly" class="dataset-thumb-empty-label">{{ label }}</span>
  </div>
</template>

<script setup lang="ts">
import { Picture } from "@element-plus/icons-vue";

defineProps({
  label: { type: String, default: "Not found" },
  /** Smaller type for list rows and field control. */
  compact: { type: Boolean, default: false },
  /** Icon only when the host box is very small. */
  iconOnly: { type: Boolean, default: false },
});
</script>

<style scoped>
.dataset-thumb-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  width: 100%;
  height: 100%;
  min-height: 0;
  padding: 4px;
  box-sizing: border-box;
  color: var(--el-text-color-placeholder);
  background: var(--el-fill-color-darker);
}
.dataset-thumb-empty-icon {
  font-size: 18px;
  opacity: 0.65;
}
.dataset-thumb-empty-label {
  font-size: 9px;
  line-height: 1.15;
  text-align: center;
  color: var(--el-text-color-secondary);
}
.dataset-thumb-empty--compact .dataset-thumb-empty-icon {
  font-size: 16px;
}
.dataset-thumb-empty--compact .dataset-thumb-empty-label {
  font-size: 8px;
}
.dataset-thumb-empty--icon-only {
  padding: 2px;
}
.dataset-thumb-empty--icon-only .dataset-thumb-empty-icon {
  font-size: 14px;
}
</style>

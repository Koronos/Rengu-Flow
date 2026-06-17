<template>
  <div class="theme-toggle">
    <el-icon class="theme-toggle__icon"><Brush /></el-icon>
    <el-switch
      :model-value="theme === 'midnight'"
      size="small"
      active-text="Cyan"
      inactive-text="Black"
      @update:model-value="(v) => setTheme(v ? 'midnight' : 'original')"
    />
  </div>
</template>

<script setup lang="ts">
import { Brush } from "@element-plus/icons-vue";
import { useTheme } from "../composables/useTheme";

const { theme, setTheme } = useTheme();
</script>

<style scoped>
.theme-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
}
.theme-toggle__icon {
  font-size: 14px;
  color: var(--el-text-color-secondary);
}
.theme-toggle :deep(.el-switch__label) {
  font-size: 11px;
}
</style>

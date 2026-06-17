<template>
  <div class="theme-toggle" role="radiogroup" aria-label="Color theme">
    <button
      v-for="opt in options"
      :key="opt.value"
      type="button"
      class="theme-swatch"
      :class="[`theme-swatch--${opt.value}`, { 'is-active': theme === opt.value }]"
      role="radio"
      :aria-checked="theme === opt.value"
      :aria-label="opt.label"
      :title="opt.label"
      @click="setTheme(opt.value)"
    />
  </div>
</template>

<script setup lang="ts">
import { useTheme, type RenguTheme } from "../composables/useTheme";

const { theme, setTheme } = useTheme();

// Each swatch previews its theme's surface + accent, so no text label is needed.
const options: { value: RenguTheme; label: string }[] = [
  { value: "midnight", label: "Midnight theme (cyan on navy)" },
  { value: "original", label: "Classic theme (black)" },
];
</script>

<style scoped>
.theme-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
}
.theme-swatch {
  width: 22px;
  height: 22px;
  border-radius: 6px;
  border: 1px solid var(--el-border-color);
  padding: 0;
  cursor: pointer;
  outline: none;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.theme-swatch:hover {
  transform: translateY(-1px);
}
.theme-swatch.is-active {
  box-shadow: 0 0 0 2px var(--el-bg-color), 0 0 0 4px var(--el-color-primary);
}
.theme-swatch:focus-visible {
  box-shadow: 0 0 0 2px var(--el-bg-color), 0 0 0 4px var(--el-color-primary);
}
/* Diagonal split: theme surface meets its accent — a tiny preview of each palette. */
.theme-swatch--midnight {
  background: linear-gradient(135deg, #0f1f33 46%, #22b8e6 46%);
}
.theme-swatch--original {
  background: linear-gradient(135deg, #1d1e1f 46%, #409eff 46%);
}
</style>

<template>
  <component
    :is="stacked ? 'article' : 'div'"
    class="dp-card"
    :class="{
      'dp-card--active': active,
      'dp-card--warning': warning,
      'dp-card--stacked': stacked,
    }"
    role="button"
    tabindex="0"
    @click="$emit('click', $event)"
    @keydown.enter="$emit('click', $event)"
  >
    <div class="dp-card-media">
      <DatasetThumbGrid :urls="displayThumbs" :fallback-text="fallbackText" flush />
    </div>
    <div class="dp-card-body">
      <div class="dp-card-head">
        <strong class="dp-card-title">{{ title }}</strong>
        <div v-if="$slots.actions" class="dp-card-actions" @click.stop>
          <slot name="actions" />
        </div>
      </div>
      <p v-if="subtitle" class="dp-card-subtitle" :title="subtitle">{{ subtitle }}</p>
      <div v-if="$slots.tags" class="dp-card-tags">
        <slot name="tags" />
      </div>
    </div>
    <el-icon v-if="active && showCheck" class="dp-card-check"><Check /></el-icon>
  </component>
</template>

<script setup>
import { computed } from "vue";
import { Check } from "@element-plus/icons-vue";
import DatasetThumbGrid from "./DatasetThumbGrid.vue";
import { usePreviewThumbs } from "../composables/usePreviewThumbs";

const props = defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: "" },
  /** Pre-resolved URLs; skipped when thumbSource is set. */
  thumbs: { type: Array, default: () => [] },
  /** { kind: 'library', id } | { kind: 'path', path } */
  thumbSource: { type: Object, default: null },
  active: { type: Boolean, default: false },
  showCheck: { type: Boolean, default: false },
  warning: { type: Boolean, default: false },
  fallbackText: { type: String, default: "…" },
  /** Directory fichas: flush media on top (no outer padding). */
  stacked: { type: Boolean, default: false },
  thumbLimit: { type: Number, default: 4 },
});

defineEmits(["click"]);

const { thumbs: loadedThumbs } = usePreviewThumbs(
  () => (props.thumbSource ? props.thumbSource : null),
  () => props.thumbLimit
);

const displayThumbs = computed(() => {
  if (props.thumbSource) return loadedThumbs.value;
  return props.thumbs || [];
});
</script>

<style scoped>
.dp-card {
  position: relative;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 10px;
  overflow: hidden;
  cursor: pointer;
  background: var(--el-fill-color-blank);
  transition:
    border-color 0.15s,
    box-shadow 0.15s,
    background 0.15s;
}
.dp-card:not(.dp-card--stacked) {
  padding: 10px;
  border-radius: var(--el-border-radius-base);
}
.dp-card:not(.dp-card--stacked) .dp-card-media {
  margin: -2px -2px 0;
}
.dp-card:hover,
.dp-card:focus-visible {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
  outline: none;
}
.dp-card--active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.dp-card--warning {
  border-color: var(--el-color-warning-light-5);
}
.dp-card--stacked .dp-card-media {
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.dp-card--warning.dp-card--stacked .dp-card-media {
  border-bottom-color: var(--el-color-warning-light-5);
  background: var(--el-color-warning-light-9);
}
.dp-card-body {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  padding: 10px 10px 12px;
}
.dp-card:not(.dp-card--stacked) .dp-card-body {
  padding: 8px 0 0;
}
.dp-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 4px;
}
.dp-card-title {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.3;
  word-break: break-word;
}
.dp-card-subtitle {
  margin: 0;
  font-size: 11px;
  color: var(--el-text-color-secondary);
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.dp-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.dp-card-check {
  position: absolute;
  top: 8px;
  right: 8px;
  color: var(--el-color-primary);
}
</style>

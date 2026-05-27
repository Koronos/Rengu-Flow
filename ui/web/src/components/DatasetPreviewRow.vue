<template>
  <div
    class="dp-row"
    :class="{ 'dp-row--warning': warning }"
    role="button"
    tabindex="0"
    @click="$emit('click', $event)"
    @keydown.enter="$emit('click', $event)"
  >
    <div class="dp-row-thumb">
      <el-image
        v-if="displayThumbs[0]"
        :src="displayThumbs[0]"
        fit="cover"
        class="dp-row-thumb-img"
        lazy
      >
        <template #error>
          <div class="dp-row-thumb-fallback" />
        </template>
      </el-image>
      <div v-else class="dp-row-thumb-fallback">{{ fallbackText }}</div>
    </div>
    <div class="dp-row-main">
      <span class="dp-row-title">{{ title }}</span>
      <span v-if="subtitle" class="dp-row-subtitle" :title="subtitle">{{ subtitle }}</span>
      <div v-if="$slots.tags" class="dp-row-tags">
        <slot name="tags" />
      </div>
    </div>
    <div v-if="$slots.actions" class="dp-row-actions" @click.stop>
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";
import { usePreviewThumbs } from "../composables/usePreviewThumbs";

const props = defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: "" },
  thumbs: { type: Array, default: () => [] },
  thumbSource: { type: Object, default: null },
  warning: { type: Boolean, default: false },
  fallbackText: { type: String, default: "…" },
  thumbLimit: { type: Number, default: 1 },
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
.dp-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  cursor: pointer;
  border-bottom: 1px solid var(--el-border-color-extra-light);
  transition: background-color 0.12s;
}
.dp-row:last-child {
  border-bottom: none;
}
.dp-row:hover,
.dp-row:focus-visible {
  background: var(--el-fill-color-light);
  outline: none;
}
.dp-row--warning {
  background: var(--el-color-warning-light-9);
}
.dp-row-thumb {
  width: 56px;
  height: 56px;
  flex-shrink: 0;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--el-border-color-lighter);
  background: var(--el-fill-color-light);
}
.dp-row-thumb-img,
.dp-row-thumb-fallback {
  width: 100%;
  height: 100%;
  display: block;
}
.dp-row-thumb-fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--el-text-color-secondary);
}
.dp-row-main {
  flex: 1;
  min-width: 0;
}
.dp-row-title {
  display: block;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
}
.dp-row-subtitle {
  display: block;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.dp-row-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}
.dp-row-actions {
  flex-shrink: 0;
}
</style>

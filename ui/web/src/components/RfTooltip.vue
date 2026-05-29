<template>
  <ElTooltip ref="tooltipRef" v-bind="mergedProps">
    <slot />
    <template v-if="$slots.content" #content>
      <slot name="content" />
    </template>
  </ElTooltip>
</template>

<script setup lang="ts">
/**
 * Safe defaults for row-action tooltips in tables/lists.
 * Element Plus does not hide the popper on trigger unmount; enterable tooltips
 * can capture the pointer and block clicks underneath.
 *
 * Props are forwarded via attrs; callers should bind tooltip options with a
 * `v-bind` record (Element Plus prop types don't expose them on a wrapper).
 */
import { ElTooltip } from "element-plus";
import type { TooltipInstance } from "element-plus";
import { computed, onBeforeUnmount, ref, useAttrs } from "vue";

defineOptions({ inheritAttrs: false });

const attrs = useAttrs();
const tooltipRef = ref<TooltipInstance>();

const mergedProps = computed(() => ({
  enterable: false,
  hideAfter: 0,
  showAfter: 300,
  teleported: true,
  popperClass: "rf-tooltip-popper",
  ...attrs,
}));

onBeforeUnmount(() => {
  tooltipRef.value?.hide?.();
});
</script>

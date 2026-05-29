<template>
  <RfTooltip v-bind="tooltipProps">
    <el-text type="info" size="small" class="path-tag">{{ path }}</el-text>
  </RfTooltip>
</template>

<script setup lang="ts">
import { computed } from "vue";
import RfTooltip from "./RfTooltip.vue";

const props = defineProps({
  path: { type: String, required: true },
});

// RfTooltip forwards options via attrs; bind them as a record so the wrapper's
// (empty) prop types don't reject `content`/`placement`/`show-after`.
const tooltipProps = computed<Record<string, unknown>>(() => ({
  content: props.path,
  placement: "top",
  "show-after": 400,
}));
</script>

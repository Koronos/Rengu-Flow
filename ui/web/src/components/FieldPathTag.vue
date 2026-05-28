<template>
  <RfTooltip :content="path" placement="top" :show-after="400">
    <el-text type="info" size="small" class="path-tag">{{ path }}</el-text>
  </RfTooltip>
</template>

<script setup lang="ts">
import RfTooltip from "./RfTooltip.vue";

defineProps({
  path: { type: String, required: true },
});
</script>

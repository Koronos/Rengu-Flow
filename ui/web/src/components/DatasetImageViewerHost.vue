<template>
  <el-image-viewer
    v-if="viewerOpen"
    :key="viewerIndex"
    :url-list="viewerUrls"
    :initial-index="viewerIndex"
    teleported
    :z-index="viewerZIndex"
    @close="closeDatasetImageViewer"
  />
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import { ElImageViewer, useZIndex } from "element-plus";
import { useDatasetImageViewer } from "../composables/useDatasetImageViewer";

const { viewerOpen, viewerUrls, viewerIndex, closeDatasetImageViewer } =
  useDatasetImageViewer();
const { nextZIndex } = useZIndex();
const viewerZIndex = ref(3001);

watch(viewerOpen, (open) => {
  if (open) viewerZIndex.value = nextZIndex();
});
</script>

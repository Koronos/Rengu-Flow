<template>
  <el-image-viewer
    v-if="viewerOpen"
    :key="viewerIndex"
    :url-list="viewerUrls"
    :initial-index="viewerIndex"
    teleported
    hide-on-click-modal
    :z-index="viewerZIndex"
    @close="closeDatasetImageViewer"
  />
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { ElImageViewer, useZIndex } from "element-plus";
import { useDatasetImageViewerStore } from "../stores/datasetImageViewer";

const viewer = useDatasetImageViewerStore();
const { viewerOpen, viewerUrls, viewerIndex } = storeToRefs(viewer);
const { closeDatasetImageViewer } = viewer;
const { nextZIndex } = useZIndex();
const viewerZIndex = ref(3001);

watch(viewerOpen, (open) => {
  if (open) viewerZIndex.value = nextZIndex();
});
</script>

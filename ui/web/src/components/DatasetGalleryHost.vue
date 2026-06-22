<template>
  <DatasetGalleryDialog
    v-model="galleryOpen"
    :title="galleryTitle"
    :content="galleryContent"
    :directory-index="galleryDirectoryIndex"
    :loading="galleryLoading"
    :error="galleryError"
  />
</template>

<script setup lang="ts">
import { watch } from "vue";
import { storeToRefs } from "pinia";
import DatasetGalleryDialog from "./DatasetGalleryDialog.vue";
import { useDatasetGalleryStore } from "../stores/datasetGallery";
import { useDatasetImageViewerStore } from "../stores/datasetImageViewer";

const {
  galleryOpen,
  galleryTitle,
  galleryContent,
  galleryDirectoryIndex,
  galleryLoading,
  galleryError,
} = storeToRefs(useDatasetGalleryStore());
const { closeDatasetImageViewer } = useDatasetImageViewerStore();

watch(galleryOpen, (open) => {
  if (!open) closeDatasetImageViewer();
});
</script>

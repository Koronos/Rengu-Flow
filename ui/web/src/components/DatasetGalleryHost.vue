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
import DatasetGalleryDialog from "./DatasetGalleryDialog.vue";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import { useDatasetImageViewer } from "../composables/useDatasetImageViewer";

const {
  galleryOpen,
  galleryTitle,
  galleryContent,
  galleryDirectoryIndex,
  galleryLoading,
  galleryError,
} = useDatasetGallery();
const { closeDatasetImageViewer } = useDatasetImageViewer();

watch(galleryOpen, (open) => {
  if (!open) closeDatasetImageViewer();
});
</script>

<template>
  <el-dialog
    :model-value="modelValue"
    :title="title"
    class="dataset-gallery-dialog"
    width="min(1200px, 96vw)"
    top="4vh"
    append-to-body
    align-center
    destroy-on-close
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <div v-loading="loading" class="dataset-gallery-dialog__body">
      <DatasetImageGallery :content="content" :directory-index="directoryIndex" expanded />
    </div>
  </el-dialog>
</template>

<script setup lang="ts">
import { ElLoadingDirective } from "element-plus";
import DatasetImageGallery from "./DatasetImageGallery.vue";

const vLoading = ElLoadingDirective;

defineProps({
  modelValue: { type: Boolean, default: false },
  title: { type: String, default: "Image gallery" },
  content: { type: String, default: "" },
  /** When set, only images from that [[directory]] index are shown. */
  directoryIndex: { type: Number, default: null },
  loading: { type: Boolean, default: false },
});

defineEmits(["update:modelValue"]);
</script>

<style scoped>
.dataset-gallery-dialog__body {
  min-height: 280px;
  max-height: calc(92vh - 7rem);
}
</style>

<style>
.dataset-gallery-dialog .el-dialog__body {
  padding-top: 8px;
}
</style>

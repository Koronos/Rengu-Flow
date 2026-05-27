<template>
  <el-dialog
    :model-value="modelValue"
    :title="title"
    width="92%"
    style="max-width: 720px"
    destroy-on-close
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <DatasetImageGallery :content="content" :directory-index="directoryIndex" />
  </el-dialog>
</template>

<script setup>
import DatasetImageGallery from "./DatasetImageGallery.vue";

defineProps({
  modelValue: { type: Boolean, default: false },
  title: { type: String, default: "Image gallery" },
  content: { type: String, default: "" },
  /** When set, only images from that [[directory]] index are shown. */
  directoryIndex: { type: Number, default: null },
});

defineEmits(["update:modelValue"]);
</script>

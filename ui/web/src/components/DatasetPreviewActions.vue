<template>
  <div class="dp-actions" @click.stop>
    <el-button
      v-if="showGallery"
      link
      :icon="Picture"
      title="Image gallery"
      :disabled="galleryDisabled"
      @click="$emit('gallery')"
    />
    <el-button
      v-if="showDelete"
      link
      type="danger"
      :icon="Delete"
      :title="deleteTitle"
      @click="$emit('delete')"
    />
  </div>
</template>

<script setup lang="ts">
import { Delete, Picture } from "@element-plus/icons-vue";

defineProps({
  showGallery: { type: Boolean, default: true },
  showDelete: { type: Boolean, default: true },
  galleryDisabled: { type: Boolean, default: false },
  deleteTitle: { type: String, default: "Remove" },
});

defineEmits(["gallery", "delete"]);
</script>

<style scoped>
.dp-actions {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  flex-shrink: 0;
}
</style>

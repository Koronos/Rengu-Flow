<template>
  <div class="dp-actions dataset-directory-actions" @click.stop>
    <el-tooltip v-if="showGallery" content="Image gallery" :show-after="300">
      <el-button
        size="small"
        circle
        :icon="Picture"
        :disabled="galleryDisabled"
        @click="$emit('gallery')"
      />
    </el-tooltip>
    <slot />
    <el-tooltip v-if="showDelete" :content="deleteTitle" :show-after="300">
      <el-button
        size="small"
        circle
        type="danger"
        :icon="Delete"
        @click="$emit('delete')"
      />
    </el-tooltip>
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
  gap: var(--rf-space-xs);
  flex-shrink: 0;
}
</style>

<template>
  <div class="library-row-crud" @click.stop>
    <el-tooltip content="Duplicate" :show-after="300">
      <el-button
        size="small"
        circle
        :icon="CopyDocument"
        :loading="loading"
        aria-label="Duplicate"
        @click="$emit('duplicate')"
      />
    </el-tooltip>
    <el-tooltip content="Delete" :show-after="300">
      <el-button
        size="small"
        circle
        type="danger"
        :icon="Delete"
        :loading="loading"
        aria-label="Delete"
        @click="$emit('delete')"
      />
    </el-tooltip>
  </div>
</template>

<script setup lang="ts">
import { CopyDocument, Delete } from "@element-plus/icons-vue";

defineProps({
  loading: { type: Boolean, default: false },
});

defineEmits<{
  duplicate: [];
  delete: [];
}>();
</script>

<style scoped>
.library-row-crud {
  display: inline-flex;
  align-items: center;
  gap: var(--rf-space-xs);
  flex-shrink: 0;
}
</style>

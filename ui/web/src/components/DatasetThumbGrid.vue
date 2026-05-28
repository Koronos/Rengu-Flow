<template>
  <div class="dataset-thumb-grid" :class="{ 'dataset-thumb-grid--flush': flush }">
    <div v-for="slot in 4" :key="slot" class="dataset-thumb-grid-cell">
      <PreviewImage
        v-if="urls[slot - 1]"
        :src="urls[slot - 1]"
        class="dataset-thumb-grid-img"
      >
        <template #error>
          <DatasetThumbEmptySlot :label="emptyLabel" />
        </template>
      </PreviewImage>
      <DatasetThumbEmptySlot v-else :label="emptyLabel" />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from "vue";
import DatasetThumbEmptySlot from "./DatasetThumbEmptySlot.vue";
import PreviewImage from "./PreviewImage.vue";

defineProps({
  urls: { type: Array as PropType<string[]>, default: () => [] },
  emptyLabel: { type: String, default: "Not found" },
  /** No bottom margin (embedded in directory cards). */
  flush: { type: Boolean, default: false },
});
</script>

<style scoped>
.dataset-thumb-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 4px;
  margin-bottom: 8px;
}
.dataset-thumb-grid--flush {
  margin-bottom: 0;
}
.dataset-thumb-grid-cell {
  aspect-ratio: 1;
  border-radius: 4px;
  overflow: hidden;
  background: var(--el-fill-color-darker);
}
.dataset-thumb-grid-img {
  width: 100%;
  height: 100%;
  display: block;
}
</style>

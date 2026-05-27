<template>
  <div class="dataset-thumb-grid" :class="{ 'dataset-thumb-grid--flush': flush }">
    <el-image
      v-for="(url, i) in urls"
      :key="i"
      :src="url"
      fit="cover"
      class="dataset-thumb-grid-cell"
      lazy
    >
      <template #error>
        <div class="dataset-thumb-grid-fallback" />
      </template>
    </el-image>
    <div
      v-if="!urls.length"
      class="dataset-thumb-grid-fallback dataset-thumb-grid-fallback--solo"
    >
      {{ fallbackText }}
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps({
  urls: { type: Array, default: () => [] },
  fallbackText: { type: String, default: "DS" },
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
.dataset-thumb-grid-cell,
.dataset-thumb-grid-fallback {
  aspect-ratio: 1;
  border-radius: 4px;
  overflow: hidden;
  background: var(--el-fill-color-darker);
}
.dataset-thumb-grid-fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--el-text-color-secondary);
}
.dataset-thumb-grid-fallback--solo {
  grid-column: span 2;
  aspect-ratio: 2 / 1;
}
</style>

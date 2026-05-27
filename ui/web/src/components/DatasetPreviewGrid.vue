<template>
  <div
    class="dp-grid"
    :class="{ 'dp-grid--scroll': scrollable, 'dp-grid--dense': dense }"
    role="list"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
defineProps({
  scrollable: { type: Boolean, default: false },
  /** Slightly smaller cards (e.g. [[directory]] tab). */
  dense: { type: Boolean, default: false },
});
</script>

<style scoped>
.dp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}
.dp-grid--dense {
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 14px;
}
@media (min-width: 1200px) {
  .dp-grid--dense {
    grid-template-columns: repeat(auto-fill, minmax(188px, 1fr));
  }
}
.dp-grid--scroll {
  max-height: min(68vh, 680px);
  overflow-y: auto;
  padding: 2px 4px 2px 2px;
}
</style>

<template>
  <div class="dp-list" :class="{ 'dp-list--scroll': scrollable }" role="list">
    <slot />
  </div>
</template>

<script setup>
defineProps({
  scrollable: { type: Boolean, default: false },
});
</script>

<style scoped>
.dp-list {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  overflow: hidden;
  background: var(--el-fill-color-blank);
}
.dp-list--scroll {
  max-height: min(68vh, 680px);
  overflow-y: auto;
}
</style>

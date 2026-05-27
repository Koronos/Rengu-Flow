<template>
  <div v-if="variant === 'library'" class="page-top library-bar">
    <slot />
    <div class="library-actions">
      <slot name="actions" />
    </div>
  </div>
  <div v-else class="editor-header-row">
    <span class="editor-title"><slot name="title" /></span>
    <div class="editor-header-main">
      <div class="editor-header-actions">
        <slot name="actions" />
      </div>
      <div v-if="$slots.trailing" class="editor-header-trailing">
        <slot name="trailing" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps({
  variant: {
    type: String,
    default: "library",
    validator: (v) => v === "library" || v === "editor",
  },
});
</script>

<style scoped>
.editor-title {
  font-weight: 600;
  min-width: 0;
  flex: 1 1 120px;
}
.editor-header-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
  min-width: 0;
  flex: 2 1 280px;
}
.editor-header-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.editor-header-trailing {
  flex-shrink: 0;
}
</style>

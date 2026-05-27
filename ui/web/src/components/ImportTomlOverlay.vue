<template>
  <div ref="rootEl" class="import-toml-overlay-root">
    <input
      ref="fileInput"
      type="file"
      accept=".toml"
      hidden
      @change="onFileInput"
    />
    <slot />
    <Teleport to="body">
      <div
        v-if="dragActive"
        class="import-overlay"
        @dragover.prevent
        @drop.prevent="onDrop"
      >
        <div class="import-overlay-inner">
          <p>Drop a .toml file to import</p>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";

const emit = defineEmits(["import"]);

const rootEl = ref<HTMLElement | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const dragActive = ref(false);
let dragDepth = 0;

function isTomlFile(file) {
  return file && (file.name.endsWith(".toml") || file.type === "application/toml");
}

function emitFile(file) {
  if (!isTomlFile(file)) return;
  emit("import", file);
}

function onDragEnter(e) {
  if (!hasToml(e)) return;
  dragDepth += 1;
  dragActive.value = true;
}

function onDragLeave() {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dragActive.value = false;
}

function onDrop(e) {
  dragDepth = 0;
  dragActive.value = false;
  const file = e.dataTransfer?.files?.[0];
  if (file) emitFile(file);
}

function hasToml(e) {
  const types = e.dataTransfer?.types || [];
  return types.includes("Files");
}

function onFileInput(e) {
  const file = e.target.files?.[0];
  if (file) emitFile(file);
  e.target.value = "";
}

function openFilePicker() {
  fileInput.value?.click();
}

onMounted(() => {
  const el = rootEl.value;
  if (!el) return;
  el.addEventListener("dragenter", onDragEnter);
  el.addEventListener("dragleave", onDragLeave);
  el.addEventListener("dragover", (e) => {
    if (hasToml(e)) e.preventDefault();
  });
  el.addEventListener("drop", onDrop);
});

onUnmounted(() => {
  const el = rootEl.value;
  if (!el) return;
  el.removeEventListener("dragenter", onDragEnter);
  el.removeEventListener("dragleave", onDragLeave);
});

defineExpose({ openFilePicker });
</script>

<style scoped>
.import-toml-overlay-root {
  display: block;
  min-height: 100%;
}
</style>

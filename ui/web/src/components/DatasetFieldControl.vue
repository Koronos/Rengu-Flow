<template>
  <div class="dataset-field">
    <div v-if="modelValue" class="dataset-field-row">
      <el-image
        v-if="thumbUrl"
        :src="thumbUrl"
        fit="cover"
        class="dataset-field-thumb"
        lazy
      >
        <template #error>
          <div class="dataset-field-thumb-fallback" />
        </template>
      </el-image>
      <div class="dataset-field-meta">
        <code>{{ displayLabel }}</code>
      </div>
      <el-button link type="danger" @click="emit('update:modelValue', '')">Clear</el-button>
    </div>
    <el-space wrap>
      <el-button @click="pickerOpen = true">Choose dataset…</el-button>
      <el-input
        :model-value="modelValue"
        placeholder="Or type a path to a .toml file"
        clearable
        class="field-path"
        @update:model-value="emit('update:modelValue', $event ?? '')"
      />
    </el-space>
    <DatasetPickerModal
      v-model="pickerOpen"
      :selected="modelValue"
      @select="onSelect"
    />
  </div>
</template>

<script setup>
import { ref, watch } from "vue";
import { api } from "../api";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import { parseDatasetLibraryRef } from "../lib/datasetLibraryRef";

const props = defineProps({
  modelValue: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const pickerOpen = ref(false);
const thumbUrl = ref("");
const displayLabel = ref("");

function parseLibraryId(path) {
  const p = parseDatasetLibraryRef(path);
  return p.isRef && p.id && /^\d+$/.test(p.id) ? p.id : null;
}

async function loadThumb(path) {
  const libraryId = parseLibraryId(path);
  const parsed = parseDatasetLibraryRef(path);
  displayLabel.value = parsed.isRef ? parsed.label || parsed.id || path : path;
  if (!libraryId) {
    thumbUrl.value = "";
    return;
  }
  try {
    const { content } = await api.getDataset(libraryId);
    const r = await api.listDatasetPreviewImages({ content, limit: 1, offset: 0 });
    const img = r.images?.[0];
    thumbUrl.value = img ? api.datasetPreviewImageUrl(img.token) : "";
  } catch {
    thumbUrl.value = "";
  }
}

function onSelect(path) {
  emit("update:modelValue", path);
}

watch(
  () => props.modelValue,
  (v) => loadThumb(v),
  { immediate: true }
);
</script>

<style scoped>
.dataset-field-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.dataset-field-thumb,
.dataset-field-thumb-fallback {
  width: 40px;
  height: 40px;
  border-radius: 4px;
  flex-shrink: 0;
  background: var(--el-fill-color-darker);
}
.dataset-field-meta {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  word-break: break-all;
}
</style>

<template>
  <div class="dataset-field">
    <div v-if="modelValue" class="dataset-field-row">
      <PreviewImage
        v-if="thumbUrl"
        :src="thumbUrl"
        class="dataset-field-thumb"
      >
        <template #error>
          <DatasetThumbEmptySlot icon-only />
        </template>
      </PreviewImage>
      <DatasetThumbEmptySlot v-else-if="modelValue" icon-only class="dataset-field-thumb" />
      <div class="dataset-field-meta">
        <code>{{ displayLabel }}</code>
      </div>
      <el-button link type="danger" @click="emit('update:modelValue', '')">Clear</el-button>
    </div>
    <el-space wrap>
      <el-button @click="pickerOpen = true">Choose dataset…</el-button>
      <PathFieldControl
        :model-value="modelValue"
        placeholder="Or type a path to a .toml file"
        expect="file"
        @update:model-value="emit('update:modelValue', $event ?? '')"
      />
    </el-space>
    <DatasetPickerModal
      v-model="pickerOpen"
      :multiple="false"
      :selected="modelValue"
      @select="onSelect"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import DatasetThumbEmptySlot from "./DatasetThumbEmptySlot.vue";
import PreviewImage from "./PreviewImage.vue";
import PathFieldControl from "./PathFieldControl.vue";
import { libraryDatasetIdFromRef } from "../lib/datasetLibraryRef";
import { peekDatasetDisplayLabel, resolveDatasetDisplayLabel } from "../lib/resolveDatasetLabels";
import { libraryThumbSource, loadPreviewThumbs } from "../lib/previewThumbs";

const props = defineProps({
  modelValue: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const pickerOpen = ref(false);
const thumbUrl = ref("");
const displayLabel = ref("");

async function loadThumb(path: string) {
  if (!path.trim()) {
    displayLabel.value = "";
    thumbUrl.value = "";
    return;
  }
  displayLabel.value = peekDatasetDisplayLabel(path);
  resolveDatasetDisplayLabel(path).then((label) => {
    if (props.modelValue === path) displayLabel.value = label;
  });
  const libraryId = libraryDatasetIdFromRef(path);
  if (!libraryId) {
    thumbUrl.value = "";
    return;
  }
  try {
    const urls = await loadPreviewThumbs(libraryThumbSource(libraryId), 1);
    thumbUrl.value = urls[0] || "";
  } catch {
    thumbUrl.value = "";
  }
}

function onSelect(path: string) {
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
.dataset-field-thumb {
  width: 40px;
  height: 40px;
  border-radius: 4px;
  flex-shrink: 0;
  overflow: hidden;
  background: var(--el-fill-color-darker);
}
.dataset-field-meta {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  word-break: break-all;
}
</style>

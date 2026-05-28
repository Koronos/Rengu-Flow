<template>
  <div class="training-datasets-field">
    <el-tag
      v-for="(entry, idx) in entries"
      :key="idx"
      closable
      class="dataset-tag"
      @close="removeAt(idx)"
    >
      {{ entryLabel(entry) }}
    </el-tag>
    <template v-if="!entries.length">
      <el-empty description="No training datasets yet" :image-size="56">
        <el-button type="primary" :icon="Plus" @click="pickerOpen = true">Add dataset</el-button>
      </el-empty>
      <PathFieldControl
        v-model="pathDraft"
        placeholder="Or type a .toml path, Enter to add"
        expect="file"
        input-class="path-draft path-draft--empty"
        @enter="addDraftPath"
      />
    </template>
    <el-space v-else wrap class="training-datasets-field__row">
      <el-button size="small" @click="pickerOpen = true">Add dataset…</el-button>
      <PathFieldControl
        v-model="pathDraft"
        placeholder="Or type a .toml path, Enter to add"
        expect="file"
        input-class="path-draft"
        @enter="addDraftPath"
      />
    </el-space>
    <p class="field-hint">
      One path uses a single dataset TOML; several paths are merged (all <code>[[directory]]</code>
      blocks) at train time, like composing in the Datasets library.
    </p>
    <DatasetPickerModal
      v-model="pickerOpen"
      multiple
      :selected="entries"
      @select-multiple="onAddMultiple"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { Plus } from "@element-plus/icons-vue";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import PathFieldControl from "./PathFieldControl.vue";
import {
  appendUniqueDatasetPaths,
  coerceTrainingDatasetEntries,
  trainingDatasetFormValue,
} from "../lib/datasetLibraryRef";
import { useResolvedDatasetLabels } from "../composables/useResolvedDatasetLabels";
import type { PropType } from "vue";

const props = defineProps({
  modelValue: { type: [Array, String] as PropType<string | string[]>, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const pickerOpen = ref(false);
const pathDraft = ref("");

const entries = computed(() => coerceTrainingDatasetEntries(props.modelValue));

const { labelFor } = useResolvedDatasetLabels(entries);

function entryLabel(entry: string): string {
  return labelFor(entry);
}

function emitPaths(paths: string[]): void {
  emit("update:modelValue", trainingDatasetFormValue(paths));
}

function removeAt(idx: number): void {
  emitPaths(entries.value.filter((_, i) => i !== idx));
}

function onAddMultiple(paths: string[]): void {
  emitPaths(appendUniqueDatasetPaths(entries.value, paths));
}

function addDraftPath() {
  const p = pathDraft.value?.trim();
  if (!p) return;
  onAddMultiple([p]);
  pathDraft.value = "";
}
</script>

<style scoped>
.training-datasets-field__row {
  width: 100%;
  max-width: 100%;
}
.dataset-tag {
  margin: 0 6px 6px 0;
}
.path-draft {
  flex: 1 1 180px;
  min-width: 0;
  max-width: 100%;
}
.path-draft--empty {
  display: block;
  margin-top: 8px;
  max-width: 100%;
}
.field-hint {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.4;
  word-break: break-word;
}
</style>

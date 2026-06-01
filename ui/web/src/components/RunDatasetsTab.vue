<template>
  <div class="run-datasets-tab">
    <p class="tab-desc">
      Pick one or more dataset TOMLs from the library (or type a path). Several datasets are merged
      (all <code>[[directory]]</code> blocks) at train time.
    </p>
    <TrainingDatasetsField :model-value="datasetValue" @update:model-value="onUpdate" />
    <div class="run-datasets-tab__actions">
      <el-button text type="primary" @click="openNewDataset">Create a new dataset…</el-button>
      <el-text type="info" size="small">Opens the dataset editor; the new dataset is added here on save.</el-text>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import TrainingDatasetsField from "./TrainingDatasetsField.vue";
import { useConfigEditorStore } from "../stores/configEditor";
import { useDatasetFormModal } from "../composables/useDatasetFormModal";
import {
  appendUniqueDatasetPaths,
  coerceTrainingDatasetEntries,
  trainingDatasetFormValue,
} from "../lib/datasetLibraryRef";

const editor = useConfigEditorStore();
const datasetModal = useDatasetFormModal();
const { form } = storeToRefs(editor);

const datasetValue = computed<string | string[]>(
  () => (form.value?.dataset as string | string[] | undefined) ?? ""
);

function onUpdate(value: string | string[]): void {
  editor.patchFormField("dataset", value);
}

function openNewDataset(): void {
  datasetModal.openCreate({
    onSaved: ({ ref }) => {
      const entries = coerceTrainingDatasetEntries(form.value?.dataset);
      onUpdate(trainingDatasetFormValue(appendUniqueDatasetPaths(entries, [ref])));
    },
  });
}
</script>

<style scoped>
.tab-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.run-datasets-tab__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  flex-wrap: wrap;
}
</style>

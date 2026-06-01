<template>
  <div class="run-datasets-tab">
    <p class="tab-desc">
      Pick one or more dataset TOMLs from the library. Several datasets are merged (all
      <code>[[directory]]</code> blocks) at train time.
    </p>

    <div class="run-datasets-tab__toolbar">
      <el-button type="primary" :icon="Plus" @click="pickerOpen = true">Add datasets</el-button>
      <el-text v-if="entries.length" type="info" size="small">
        {{ entries.length }} {{ entries.length === 1 ? "dataset" : "datasets" }} selected
      </el-text>
    </div>

    <DatasetPreviewCollection
      v-if="entries.length"
      :items="previewItems"
      view-mode="cards"
      class="run-datasets-tab__selected"
    >
      <template #actions="{ item }">
        <DatasetPreviewActions
          :gallery-disabled="!item.libraryId"
          delete-title="Remove from run"
          @gallery="openGallery(item)"
          @delete="removeEntry(item.path)"
        />
      </template>
    </DatasetPreviewCollection>
    <el-empty v-else description="No datasets selected yet" :image-size="56">
      <el-button type="primary" :icon="Plus" @click="pickerOpen = true">Add datasets</el-button>
    </el-empty>

    <el-collapse class="run-datasets-tab__advanced">
      <el-collapse-item title="Add a raw .toml path" name="path">
        <PathFieldControl
          v-model="pathDraft"
          placeholder="Type a .toml path, Enter to add"
          expect="file"
          input-class="run-datasets-tab__path"
          @enter="addDraftPath"
        />
        <p class="field-hint">
          For datasets not in the library. Library datasets are easier to preview and reuse.
        </p>
      </el-collapse-item>
    </el-collapse>

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
import { storeToRefs } from "pinia";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import DatasetPreviewActions from "./DatasetPreviewActions.vue";
import DatasetPreviewCollection from "./DatasetPreviewCollection.vue";
import PathFieldControl from "./PathFieldControl.vue";
import { useConfigEditorStore } from "../stores/configEditor";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import {
  useSelectedDatasetPreviews,
  type SelectedDatasetPreviewItem,
} from "../composables/useSelectedDatasetPreviews";
import {
  appendUniqueDatasetPaths,
  canonicalDatasetRef,
  coerceTrainingDatasetEntries,
  trainingDatasetFormValue,
} from "../lib/datasetLibraryRef";

const editor = useConfigEditorStore();
const { form } = storeToRefs(editor);
const { showFromLibrary } = useDatasetGallery();

const pickerOpen = ref(false);
const pathDraft = ref("");

const entries = computed(() => coerceTrainingDatasetEntries(form.value?.dataset));
const { previewItems } = useSelectedDatasetPreviews(entries);

function emitEntries(next: string[]): void {
  editor.patchFormField("dataset", trainingDatasetFormValue(next));
}

function onAddMultiple(paths: string[]): void {
  emitEntries(appendUniqueDatasetPaths(entries.value, paths));
}

function removeEntry(path: string): void {
  const key = canonicalDatasetRef(path);
  emitEntries(entries.value.filter((entry) => canonicalDatasetRef(entry) !== key));
}

function addDraftPath(): void {
  const p = pathDraft.value?.trim();
  if (!p) return;
  onAddMultiple([p]);
  pathDraft.value = "";
}

function openGallery(item: SelectedDatasetPreviewItem): void {
  if (!item.libraryId) return;
  showFromLibrary({
    id: item.libraryId,
    title: `Gallery — ${item.title}`,
    directoryIndex: null,
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
.run-datasets-tab__toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.run-datasets-tab__selected {
  margin-bottom: 12px;
}
.run-datasets-tab__advanced {
  margin-top: 4px;
  border-top: none;
}
.run-datasets-tab__path {
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

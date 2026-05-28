<template>
  <el-dialog
    v-model="visible"
    class="dataset-folder-dialog"
    :title="isEdit ? 'Edit [[directory]]' : 'Add [[directory]]'"
    width="min(600px, 96vw)"
    destroy-on-close
    @closed="onClosed"
  >
    <div class="folder-dialog-layout">
      <el-form label-position="top" class="folder-dialog-primary">
        <el-card shadow="never" class="section-card">
          <template #header>
            <span class="section-title">This directory only</span>
          </template>
          <p class="section-desc">
            Identity of this <code>[[directory]]</code> row: folder path, epoch repeats, and caption
            prefix. These are not inherited from the dataset defaults tab.
          </p>
          <ConfigFormField
            v-for="field in primaryFields"
            :key="field.path"
            :field="field"
            :form="draft"
            dataset-form
            @update:path="onField"
          />
        </el-card>
      </el-form>

      <el-card shadow="never" class="section-card folder-dialog-overrides">
        <template #header>
          <span class="section-title">Overrides global defaults</span>
        </template>
        <p class="section-desc">
          Optional. Unset fields use values from the <strong>Dataset defaults</strong> tab (TOML root).
          Turn on a switch or change a control to write an override on this row; matching values are
          omitted when saving.
        </p>

        <template v-if="explicitOverrideFields.length">
          <h4 class="subsection-title">Per-directory behavior</h4>
          <p class="subsection-desc">
            Caching and caption options for this folder. Leave unchanged to inherit dataset defaults.
          </p>
          <div
            v-for="field in explicitOverrideFields"
            :key="field.path"
            class="override-block"
          >
            <DirectoryOverrideFieldBlock
              :field="field"
              :entry="draft"
              :global-form="globalForm"
              @update:enabled="(on) => setOverrideOn(field, on)"
              @update:path="onField"
            />
          </div>
        </template>

        <template v-if="rootOverrideFields.length">
          <h4 class="subsection-title">Replace root-level settings</h4>
          <p class="subsection-desc">
            Same keys as on the dataset defaults tab (e.g. <code>resolutions</code>,
            <code>frame_buckets</code>). Enable to override for this folder only.
          </p>
          <div
            v-for="field in rootOverrideFields"
            :key="field.path"
            class="override-block"
          >
            <DirectoryOverrideFieldBlock
              :field="field"
              :entry="draft"
              :global-form="globalForm"
              @update:enabled="(on) => setOverrideOn(field, on)"
              @update:path="onField"
            />
          </div>
        </template>

        <template v-if="conditionalOverrideFields.length">
          <h4 class="subsection-title">Dependent overrides</h4>
          <p class="subsection-desc">Shown when a related per-directory or override field is enabled.</p>
          <div
            v-for="field in conditionalOverrideFields"
            :key="field.path"
            class="override-block"
          >
            <DirectoryOverrideFieldBlock
              :field="field"
              :entry="draft"
              :global-form="globalForm"
              @update:enabled="(on) => setOverrideOn(field, on)"
              @update:path="onField"
            />
          </div>
        </template>
      </el-card>
    </div>

    <template #footer>
      <el-button @click="visible = false">Cancel</el-button>
      <el-button type="primary" :disabled="!canSave" @click="confirm">
        {{ isEdit ? "Apply" : "Add" }}
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import DirectoryOverrideFieldBlock from "./DirectoryOverrideFieldBlock.vue";
import {
  conditionalDirectoryOverrideFields,
  emptyDirectoryRow,
  explicitDirectoryOverrideFields,
  optionalRootOverrideFields,
  overrideDirectoryFields,
  primaryDirectoryFields,
  setOverrideEnabled,
} from "../lib/datasetDirectoryForm";
import { clonePlain } from "../lib/clonePlain";
import { useDatasetEditorStore } from "../stores/datasetEditor";
import type { FormValues, SchemaField } from "../types/forms";
import type { DirectoryFormRow } from "../lib/datasetDirectoryForm";

interface DatasetFolderDialogProps {
  modelValue: boolean;
  schema: { directory_fields?: SchemaField[] } | null;
  entry: DirectoryFormRow | null;
  editIndex: number;
}

const props = withDefaults(defineProps<DatasetFolderDialogProps>(), {
  modelValue: false,
  schema: null,
  entry: null,
  editIndex: -1,
});

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "save", payload: { entry: DirectoryFormRow; index: number }): void;
}>();

const editor = useDatasetEditorStore();
const { form: globalFormRef } = storeToRefs(editor);

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit("update:modelValue", v),
});

const draft = ref<DirectoryFormRow>(emptyDirectoryRow());

const isEdit = computed(() => props.editIndex >= 0);
const primaryFields = computed(() => primaryDirectoryFields(props.schema));
const overrideFields = computed(() => overrideDirectoryFields(props.schema));
const explicitOverrideFields = computed(() =>
  explicitDirectoryOverrideFields(overrideFields.value)
);
const rootOverrideFields = computed(() =>
  optionalRootOverrideFields(overrideFields.value)
);
const conditionalOverrideFields = computed(() =>
  conditionalDirectoryOverrideFields(overrideFields.value)
);

const globalForm = computed<FormValues>(() => globalFormRef.value ?? {});

/** New entries need a path; edits may clear it (shown as not found). */
const canSave = computed(
  () => isEdit.value || (draft.value.path || "").trim().length > 0
);

watch(
  () => [props.modelValue, props.entry],
  () => {
    if (!props.modelValue) return;
    draft.value = props.entry ? clonePlain(props.entry) : emptyDirectoryRow();
  },
  { immediate: true }
);

function onField({ path, value }: { path: string; value: unknown }) {
  draft.value = { ...draft.value, [path]: value };
}

function setOverrideOn(field: SchemaField, on: boolean) {
  draft.value = setOverrideEnabled(field, draft.value, on);
}

function confirm() {
  if (!canSave.value) return;
  const row = { ...draft.value };
  row.path = row.path.trim();
  row.num_repeats = Number(row.num_repeats) || 1;
  emit("save", { entry: row, index: props.editIndex });
  visible.value = false;
}

function onClosed() {
  draft.value = emptyDirectoryRow();
}
</script>

<style scoped>
.folder-dialog-layout {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.folder-dialog-primary {
  flex-shrink: 0;
}
.folder-dialog-overrides {
  flex-shrink: 0;
  border: 1px solid var(--el-border-color-lighter);
}
.section-card {
  border: 1px solid var(--el-border-color-lighter);
}
.section-title {
  font-weight: 600;
}
.section-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.section-desc code {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.subsection-title {
  margin: 4px 0 4px;
  font-size: 13px;
  font-weight: 600;
}
.subsection-desc {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.4;
}
.subsection-desc code {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
}
.override-block {
  margin-bottom: 10px;
}
</style>

<style>
.dataset-folder-dialog .el-dialog__body {
  max-height: min(75vh, 720px);
  overflow-y: auto;
  padding-top: 8px;
}
</style>

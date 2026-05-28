<template>
  <el-dialog
    v-model="visible"
    class="preview-entry-dialog"
    :title="isEdit ? 'Edit preview configuration' : 'Add preview configuration'"
    width="min(640px, 96vw)"
    destroy-on-close
    @closed="onClosed"
  >
    <p class="dialog-intro">
      One preview configuration = one TensorBoard image tag. Use overrides only when this prompt
      needs different dimensions, schedule, or seeds than the global preview settings.
    </p>

    <el-form label-position="top" class="preview-entry-form">
      <el-card shadow="never" class="section-card">
        <template #header>
          <span class="section-title">Prompt</span>
        </template>
        <ConfigFormField
          v-for="field in primaryFields"
          :key="field.path"
          :field="field"
          :form="visibilityForm"
          :capabilities="capabilities"
          always-visible
          @update:path="onField"
        />
      </el-card>

      <el-collapse v-if="overrideFields.length" class="override-collapse">
        <el-collapse-item name="overrides">
          <template #title>
            <span class="group-title">Overrides (optional)</span>
          </template>
          <p class="section-desc">
            Unset fields inherit from global preview settings on the Previews tab.
          </p>
          <el-row :gutter="16">
            <el-col
              v-for="field in overrideFields"
              :key="field.path"
              :xs="24"
              :sm="overrideColSpan(field)"
            >
              <ConfigFormField
                :field="field"
                :form="visibilityForm"
                :capabilities="capabilities"
                always-visible
                @update:path="onField"
              />
            </el-col>
          </el-row>
        </el-collapse-item>
      </el-collapse>
    </el-form>

    <template #footer>
      <el-button @click="visible = false">Cancel</el-button>
      <el-button type="primary" :disabled="!canSave" @click="onSave">
        {{ isEdit ? "Save" : "Add" }}
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref, watch, type PropType } from "vue";
import ConfigFormField from "./ConfigFormField.vue";
import { fieldVisible } from "../lib/formUtils";
import { schemaOptionalFieldColSpan } from "../lib/schemaFieldLayout";
import {
  emptyPreviewEntryTable,
  previewEntryIsValid,
  previewEntryToDraft,
  serializePreviewEntry,
  type PreviewEntry,
  type PreviewEntryTable,
} from "../lib/previewEntries";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  entryFields: { type: Array as PropType<SchemaField[]>, default: () => [] },
  entry: { type: [String, Object] as PropType<PreviewEntry | null>, default: null },
  editIndex: { type: Number, default: -1 },
  parentForm: { type: Object as PropType<FormValues>, default: () => ({}) },
  capabilities: { type: Object as PropType<ModelCapabilities>, default: () => ({}) },
});

const emit = defineEmits<{
  "update:modelValue": [open: boolean];
  save: [payload: { entry: PreviewEntry; index: number }];
}>();

const visible = computed({
  get: () => props.modelValue,
  set: (v: boolean) => emit("update:modelValue", v),
});

const isEdit = computed(() => props.editIndex >= 0);

const draft = ref<PreviewEntryTable>(emptyPreviewEntryTable());

const primaryPaths = new Set(["name", "prompt"]);

const primaryFields = computed(() =>
  props.entryFields.filter((f) => primaryPaths.has(f.path || ""))
);

const overrideFields = computed(() =>
  props.entryFields.filter(
    (f) =>
      f.path &&
      !primaryPaths.has(f.path) &&
      fieldVisible(f, visibilityForm.value, props.capabilities)
  )
);

const visibilityForm = computed(() => ({
  ...props.parentForm,
  ...draft.value,
}));

const canSave = computed(() => previewEntryIsValid(serializePreviewEntry(draft.value)));

watch(
  () => [props.modelValue, props.entry] as const,
  ([open]) => {
    if (!open) return;
    draft.value = previewEntryToDraft(props.entry);
  },
  { immediate: true }
);

function overrideColSpan(field: SchemaField): number {
  return schemaOptionalFieldColSpan(field);
}

function onField({ path, value }: { path: string; value: unknown }): void {
  if (!path) return;
  draft.value = { ...draft.value, [path]: value };
}

function onSave(): void {
  if (!canSave.value) return;
  emit("save", { entry: serializePreviewEntry(draft.value), index: props.editIndex });
  visible.value = false;
}

function onClosed(): void {
  draft.value = emptyPreviewEntryTable();
}
</script>

<style scoped>
.dialog-intro {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.section-card {
  border: 1px solid var(--el-border-color-lighter);
  margin-bottom: 12px;
}
.section-card :deep(.el-card__header) {
  padding: 10px 14px;
}
.section-card :deep(.el-card__body) {
  padding: 10px 14px 14px;
}
.section-title {
  font-weight: 600;
  font-size: 13px;
}
.section-desc {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.override-collapse {
  border: none;
}
.override-collapse :deep(.el-collapse-item__header) {
  border: none;
  height: 36px;
}
.override-collapse :deep(.el-collapse-item__wrap) {
  border: none;
}
.group-title {
  font-weight: 500;
  color: var(--el-text-color-secondary);
}
</style>

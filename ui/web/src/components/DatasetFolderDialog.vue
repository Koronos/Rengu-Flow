<template>
  <el-dialog
    v-model="visible"
    :title="isEdit ? 'Edit [[directory]]' : 'Add [[directory]]'"
    width="min(560px, 96vw)"
    destroy-on-close
    @closed="onClosed"
  >
    <el-form label-position="top" class="folder-dialog-form" @submit.prevent>
      <ConfigFormField
        v-for="field in primaryFields"
        :key="field.path"
        :field="field"
        :form="draft"
        dataset-form
        @update:path="onField"
      />

      <el-divider content-position="left">Overrides</el-divider>
      <p class="override-hint">
        Optional per-directory settings. When unset, values inherit from the dataset defaults tab (TOML root).
      </p>

      <div v-for="field in overrideFields" :key="field.path" class="override-block">
        <div v-if="needsOverrideToggle(field)" class="override-toggle">
          <el-switch
            :model-value="overrideOn(field)"
            @update:model-value="(on) => setOverrideOn(field, on)"
          />
          <span>{{ overrideToggleLabel(field) }}</span>
        </div>
        <ConfigFormField
          v-if="showOverrideControl(field)"
          :field="field"
          :form="draft"
          dataset-form
          :always-visible="needsOverrideToggle(field)"
          @update:path="onField"
        />
      </div>
    </el-form>

    <template #footer>
      <el-button @click="visible = false">Cancel</el-button>
      <el-button type="primary" :disabled="!canSave" @click="confirm">
        {{ isEdit ? "Apply" : "Add" }}
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import ConfigFormField from "./ConfigFormField.vue";
import {
  emptyDirectoryRow,
  isOverrideEnabled,
  overrideDirectoryFields,
  primaryDirectoryFields,
  setOverrideEnabled,
} from "../lib/datasetDirectoryForm";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  schema: { type: Object, default: null },
  entry: { type: Object, default: null },
  editIndex: { type: Number, default: -1 },
});

const emit = defineEmits(["update:modelValue", "save"]);

const visible = computed({
  get: () => props.modelValue,
  set: (v) => emit("update:modelValue", v),
});

const draft = ref(emptyDirectoryRow());

const isEdit = computed(() => props.editIndex >= 0);
const primaryFields = computed(() => primaryDirectoryFields(props.schema));
const overrideFields = computed(() => overrideDirectoryFields(props.schema));

/** New entries need a path; edits may clear it (shown as not found). */
const canSave = computed(
  () => isEdit.value || (draft.value.path || "").trim().length > 0
);

watch(
  () => [props.modelValue, props.entry],
  () => {
    if (!props.modelValue) return;
    draft.value = props.entry
      ? JSON.parse(JSON.stringify(props.entry))
      : emptyDirectoryRow();
  },
  { immediate: true }
);

function onField({ path, value }) {
  draft.value = { ...draft.value, [path]: value };
}

function needsOverrideToggle(field) {
  return !!(field.show_if_set || field.show_when_field);
}

/** Always-on overrides use the control directly; optional ones need the toggle on first. */
function showOverrideControl(field) {
  if (!needsOverrideToggle(field)) return true;
  return overrideOn(field);
}

function overrideToggleLabel(field) {
  if (field.type === "boolean") {
    return `Override — ${field.label}`;
  }
  return field.label;
}

function overrideOn(field) {
  return isOverrideEnabled(field, draft.value);
}

function setOverrideOn(field, on) {
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
.folder-dialog-form {
  max-height: min(70vh, 640px);
  overflow-y: auto;
  padding-right: 4px;
}
.override-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.override-block {
  margin-bottom: 8px;
}
.override-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 4px;
  font-size: 14px;
}
</style>

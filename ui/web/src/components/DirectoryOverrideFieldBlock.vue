<template>
  <div v-if="blockVisible" class="directory-override-field">
    <div class="override-head">
      <el-tag
        :type="status === 'inherited' ? 'info' : 'warning'"
        size="small"
        effect="plain"
        class="status-tag"
      >
        {{ status === "inherited" ? "Inherited" : "Overrides global" }}
      </el-tag>
      <div v-if="needsToggle" class="override-toggle">
        <el-switch
          :model-value="toggleOn"
          @update:model-value="(on) => emit('update:enabled', Boolean(on))"
        />
        <span>{{ toggleLabel }}</span>
        <FieldHelpIcon :field="field" />
      </div>
    </div>

    <p v-if="status === 'inherited' && globalHint" class="inherit-hint">
      Uses dataset default: <span class="inherit-value">{{ globalHint }}</span>
    </p>

    <el-form
      v-if="showControl"
      label-position="top"
      class="directory-override-control-form"
    >
      <ConfigFormField
        :field="field"
        :form="entry"
        dataset-form
        :directory-inherit-form="globalForm"
        :always-visible="needsToggle"
        :hide-label="needsToggle"
        hide-label-help
        :path-tag-placement="needsToggle ? 'foot' : 'label'"
        @update:path="(payload) => emit('update:path', payload)"
      />
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import ConfigFormField from "./ConfigFormField.vue";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import {
  directoryFieldOverrideStatus,
  directoryOverrideBlockVisible,
  globalFieldDisplayHint,
  isOverrideEnabled,
  needsDirectoryOverrideToggle,
} from "../lib/datasetDirectoryForm";
import type { DirectoryFormRow } from "../lib/datasetDirectoryForm";
import type { FormValues, SchemaField } from "../types/forms";

const props = defineProps<{
  field: SchemaField;
  entry: DirectoryFormRow;
  globalForm: FormValues;
}>();

const emit = defineEmits<{
  "update:enabled": [on: boolean];
  "update:path": [payload: { path: string; value: unknown }];
}>();

const needsToggle = computed(() => needsDirectoryOverrideToggle(props.field));

const blockVisible = computed(() =>
  directoryOverrideBlockVisible(props.field, props.entry, props.globalForm)
);

const status = computed(() => directoryFieldOverrideStatus(props.field, props.entry));

const toggleOn = computed(() => isOverrideEnabled(props.field, props.entry));

const showControl = computed(() => {
  if (!needsToggle.value) return true;
  return toggleOn.value;
});

const globalHint = computed(() => globalFieldDisplayHint(props.field, props.globalForm));

function getToggleLabel(field: SchemaField): string {
  if (field.type === "boolean") {
    return `Override — ${field.label || field.path}`;
  }
  return field.label || field.path;
}

const toggleLabel = computed(() => getToggleLabel(props.field));
</script>

<style scoped>
.directory-override-field {
  border: 1px solid var(--el-border-color-extra-light);
  border-radius: 6px;
  padding: 10px 12px;
  background: var(--el-fill-color-blank);
}
.override-head {
  display: flex;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 8px 12px;
  margin-bottom: 4px;
}
.status-tag {
  flex-shrink: 0;
}
.override-toggle {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px 10px;
  font-size: 14px;
  flex: 1;
  min-width: 0;
}
.inherit-hint {
  margin: 0 0 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.inherit-value {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
}
.directory-override-field :deep(.directory-override-control-form .el-form-item) {
  margin-bottom: 0;
}
</style>

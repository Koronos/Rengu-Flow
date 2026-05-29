<template>
  <el-card shadow="never" class="section-card">
    <template #header>
      <div class="sec-header">
        <span class="section-title">{{ section.title }}</span>
        <div v-if="attentionCount" class="sec-attention">
          <span
            v-if="unfilledRequired"
            class="sec-attention-item"
            :title="`${unfilledRequired} required field(s) empty`"
          >
            <span class="rf-label-required" aria-hidden="true">*</span>
            {{ unfilledRequired }}
          </span>
        </div>
      </div>
    </template>

    <p v-if="section.description" class="sec-desc">{{ section.description }}</p>

    <template v-if="section.id === 'preview'">
      <PreviewEntriesField
        :model-value="previewPromptsValue"
        :entry-fields="previewEntryFields"
        :parent-form="formValues"
        :capabilities="capabilities"
        @update:model-value="onPreviewPromptsUpdate"
      />
      <div class="group-title">Global preview settings</div>
      <p class="sec-desc preview-global-hint">
        Defaults for all preview rows (schedule, size, seeds). Override per row in the dialog.
      </p>
    </template>

    <div v-if="adapterMode" class="adapter-mode-row">
      <ConfigFormField
        :field="adapterMode"
        :form="formValues"
        :capabilities="capabilities"
        @update:path="onFieldUpdate"
      />
    </div>

    <el-alert
      v-if="section.id === 'model' && selectedCapability"
      type="info"
      :closable="false"
      show-icon
      class="registry-alert"
    >
      <strong>{{ selectedCapability.display_name }}</strong>
      — supported training: {{ trainingModesText }}
      <template v-if="selectedCapability.branding_note">
        <br />
        <span class="branding-note">{{ selectedCapability.branding_note }}</span>
      </template>
    </el-alert>

    <el-alert
      v-if="section.id === 'adapter' && selectedCapability"
      type="info"
      :closable="false"
      show-icon
      class="registry-alert"
    >
      <template v-if="modelSupportsAdapters(selectedCapability)">
        Adapter types: <strong>{{ selectedCapability.adapters?.join(", ") }}</strong>
        <span v-if="selectedCapability.full_finetune"> — or disable adapter for full finetune.</span>
      </template>
      <template v-else-if="selectedCapability.full_finetune">
        Full-model finetune only (no LoRA / LoKr).
      </template>
    </el-alert>

    <el-form label-position="top" class="config-form">
      <template v-if="partition.required.length">
        <div class="group-title">Required</div>
        <el-row :gutter="16">
          <el-col v-for="field in partition.required" :key="field.path" :xs="24" :sm="fieldColSpan(field)">
            <ConfigFormField
              :field="field"
              :form="formValues"
              :capabilities="capabilities"
              @update:path="onFieldUpdate"
            />
          </el-col>
        </el-row>
      </template>

      <template v-if="partition.recommended.length">
        <div class="group-title group-title--important">
          <template v-if="section.id === 'preview'">Schedule &amp; toggles</template>
          <template v-else>
            Important
            <el-text type="info" size="small" class="group-hint">
              Has a default in the trainer — review before running
            </el-text>
          </template>
        </div>
        <el-row :gutter="16">
          <el-col v-for="field in partition.recommended" :key="field.path" :xs="24" :sm="fieldColSpan(field)">
            <ConfigFormField
              :field="field"
              :form="formValues"
              :capabilities="capabilities"
              @update:path="onFieldUpdate"
            />
          </el-col>
        </el-row>
      </template>

      <template v-if="partition.advanced.length">
        <div v-if="section.id === 'preview'" class="group-title optional-title">
          Generation defaults
        </div>
        <el-row :gutter="16">
          <el-col v-for="field in partition.advanced" :key="field.path" :xs="24" :sm="fieldColSpan(field)">
            <ConfigFormField
              :field="field"
              :form="formValues"
              :capabilities="capabilities"
              @update:path="onFieldUpdate"
            />
          </el-col>
        </el-row>
      </template>
    </el-form>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import PreviewEntriesField from "./PreviewEntriesField.vue";
import type { PreviewEntry } from "../lib/previewEntries";
import type { ConfigSchemaSection } from "../lib/configFormSections";
import { configFieldColSpan } from "../lib/configFormSections";
import {
  PREVIEW_PROMPTS_PATH,
  adapterModeField,
  partitionSectionFields,
  sectionAttentionCount,
  unfilledRequiredCount,
} from "../lib/configFormSectionLogic";
import { modelSupportsAdapters, trainingModesLabel } from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";
import type { FormValues, ModelCapability, SchemaField } from "../types/forms";

const props = defineProps<{
  section: ConfigSchemaSection;
  selectedCapability: ModelCapability | null;
  previewEntryFields: SchemaField[];
}>();

const editor = useConfigEditorStore();
const { form, modelCapabilities } = storeToRefs(editor);

const formValues = computed(() => form.value ?? ({} as FormValues));
const capabilities = computed(() => modelCapabilities.value);

const partition = computed(() =>
  partitionSectionFields(props.section, formValues.value, capabilities.value)
);
const attentionCount = computed(() =>
  sectionAttentionCount(props.section, formValues.value, capabilities.value)
);
const unfilledRequired = computed(() =>
  unfilledRequiredCount(props.section, formValues.value, capabilities.value)
);
const adapterMode = computed(() =>
  adapterModeField(props.section, formValues.value, capabilities.value)
);
const trainingModesText = computed(() => trainingModesLabel(props.selectedCapability));
const previewPromptsValue = computed(() => formValues.value[PREVIEW_PROMPTS_PATH]);

function fieldColSpan(field: SchemaField): number {
  return configFieldColSpan(field);
}

function onFieldUpdate({ path, value }: { path: string; value: unknown }): void {
  editor.patchFormField(path, value);
}

function onPreviewPromptsUpdate(entries: PreviewEntry[]): void {
  editor.patchFormField(PREVIEW_PROMPTS_PATH, entries.length ? entries : null);
}
</script>

<style scoped>
.section-card {
  border: 1px solid var(--el-border-color-lighter);
}
.section-card :deep(.el-card__header) {
  padding: 12px 16px;
}
.section-card :deep(.el-card__body) {
  padding: 12px 16px 16px;
}
.sec-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.section-title {
  font-weight: 600;
}
.sec-attention {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.sec-attention-item {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}
.sec-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.adapter-mode-row {
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: var(--el-border-radius-base);
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color);
}
.adapter-mode-row :deep(.el-form-item) {
  margin-bottom: 0;
}
.registry-alert {
  margin-bottom: 12px;
}
.branding-note {
  font-size: 12px;
  opacity: 0.9;
}
.config-form {
  width: 100%;
}
.config-form :deep(.el-form-item) {
  margin-bottom: 14px;
}
.group-title {
  font-size: 13px;
  font-weight: 600;
  margin: 12px 0 8px;
  color: var(--el-text-color-primary);
}
.group-title:first-child {
  margin-top: 0;
}
.group-title--important {
  color: var(--el-text-color-primary);
}
.group-hint {
  margin-left: 8px;
  font-weight: 400;
}
.optional-title {
  font-weight: 500;
  color: var(--el-text-color-secondary);
}
.preview-global-hint {
  margin-top: -4px;
}
</style>

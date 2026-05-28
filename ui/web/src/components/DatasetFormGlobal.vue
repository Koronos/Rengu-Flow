<template>
  <div class="dataset-form-global">
    <p class="tab-intro">
      Root-level dataset TOML (e.g. <code>resolutions</code>, <code>frame_buckets</code>). Each
      <code>[[directory]]</code> inherits these unless it overrides them.
    </p>
    <el-card
      v-for="sec in sections"
      :key="sec.id"
      shadow="never"
      class="section-card"
    >
      <template #header>
        <span class="section-title">{{ sec.title }}</span>
      </template>
      <p v-if="sec.description" class="section-desc">{{ sec.description }}</p>

      <el-form label-position="top" class="global-form">
        <el-row :gutter="16">
          <el-col
            v-for="field in coreFields(sec)"
            :key="field.path"
            :xs="24"
            :sm="schemaFieldColSpan(field)"
          >
            <ConfigFormField
              :field="field"
              :form="formValues"
              dataset-form
              @update:path="onField"
            />
          </el-col>
        </el-row>

        <el-row :gutter="16" class="optional-row">
          <el-col
            v-for="field in optionalFields(sec)"
            :key="field.path"
            :xs="24"
            :sm="schemaOptionalFieldColSpan(field)"
          >
            <div class="optional-block">
              <div class="optional-toggle">
                <el-switch
                  :model-value="optionalActive(field)"
                  @update:model-value="(on) => setOptionalActive(field, Boolean(on))"
                />
                <span>{{ optionalToggleLabel(field) }}</span>
                <FieldHelpIcon :field="field" />
              </div>
              <ConfigFormField
                v-if="optionalActive(field)"
                :field="field"
                :form="formValues"
                dataset-form
                always-visible
                @update:path="onField"
              />
            </div>
          </el-col>
        </el-row>

        <DatasetTagDropoutField
          v-if="sec.id === 'captions'"
          :form="formValues"
          :schema-fields="sec.fields || []"
          @update:path="(path, value) => editor.patchFormField(path, value)"
        />
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import DatasetTagDropoutField from "./DatasetTagDropoutField.vue";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import {
  isOverrideEnabled,
  sectionCoreFields,
  sectionOptionalFields,
  setOverrideEnabled,
} from "../lib/datasetFormSections";
import { schemaFieldColSpan, schemaOptionalFieldColSpan } from "../lib/schemaFieldLayout";
import { useDatasetEditorStore } from "../stores/datasetEditor";
import type { FormValues, SchemaField } from "../types/forms";

const editor = useDatasetEditorStore();
const { form, schema } = storeToRefs(editor);

interface DatasetFormSection {
  id: string;
  title?: string;
  description?: string;
  fields?: SchemaField[];
}

const TAG_DROPOUT_SCHEMA_PATHS = new Set([
  "tag_dropout_enabled",
  "tag_dropout_probability",
  "tag_dropout_mode",
  "tag_match_case_sensitive",
  "tag_dropout_rules",
  "uncond_fraction",
]);

const HIDDEN_SECTION_IDS = new Set(["augmentation_global"]);

const sections = computed(
  () =>
    ((schema.value?.sections as DatasetFormSection[] | undefined) || []).filter(
      (sec) => !HIDDEN_SECTION_IDS.has(sec.id)
    )
);

const formValues = computed<FormValues>(() => form.value ?? {});

function withoutTagDropoutFields(fields: SchemaField[]): SchemaField[] {
  return fields.filter((f) => !TAG_DROPOUT_SCHEMA_PATHS.has(f.path));
}

function coreFields(sec: DatasetFormSection): SchemaField[] {
  return withoutTagDropoutFields(sectionCoreFields(sec));
}

function optionalFields(sec: DatasetFormSection): SchemaField[] {
  return withoutTagDropoutFields(sectionOptionalFields(sec));
}

function onField({ path, value }: { path: string; value: unknown }) {
  editor.patchFormField(path, value);
}

function optionalActive(field: SchemaField): boolean {
  return isOverrideEnabled(field, formValues.value);
}

function optionalToggleLabel(field: SchemaField): string {
  if (field.type === "boolean") {
    return `Override — ${field.label || field.path}`;
  }
  return field.label || field.path;
}

function setOptionalActive(field: SchemaField, on: boolean) {
  editor.setForm(setOverrideEnabled(field, formValues.value, on));
}
</script>

<style scoped>
.dataset-form-global {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.tab-intro {
  margin: 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.tab-intro code {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
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
.global-form :deep(.el-form-item) {
  margin-bottom: 14px;
}
.optional-row {
  margin-top: 4px;
}
.optional-block {
  margin-bottom: 4px;
}
.optional-toggle {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px 10px;
  margin-bottom: 4px;
  font-size: 14px;
}
</style>

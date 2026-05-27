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
            :sm="fieldColSpan(field)"
          >
            <ConfigFormField
              :field="field"
              :form="form"
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
            :sm="optionalColSpan(field)"
          >
            <div class="optional-block">
              <div class="optional-toggle">
                <el-switch
                  :model-value="optionalActive(field)"
                  @update:model-value="(on) => setOptionalActive(field, on)"
                />
                <span>{{ optionalToggleLabel(field) }}</span>
              </div>
              <ConfigFormField
                v-if="optionalActive(field)"
                :field="field"
                :form="form"
                dataset-form
                always-visible
                @update:path="onField"
              />
            </div>
          </el-col>
        </el-row>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import {
  optionalFieldActive,
  sectionCoreFields,
  sectionOptionalFields,
  setOverrideEnabled,
} from "../lib/datasetFormSections";
import { useDatasetEditorStore } from "../stores/datasetEditor";

const editor = useDatasetEditorStore();
const { form, schema } = storeToRefs(editor);

interface DatasetFormSection {
  id: string;
  title?: string;
  description?: string;
  fields?: Record<string, unknown>[];
}

const sections = computed(
  () => (schema.value?.sections as DatasetFormSection[] | undefined) || []
);

function coreFields(sec) {
  return sectionCoreFields(sec);
}

function optionalFields(sec) {
  return sectionOptionalFields(sec);
}

function fieldColSpan(field) {
  if (field.type === "json" || field.type === "integer_list" || field.type === "number_list") {
    return 24;
  }
  if (field.type === "boolean") return 12;
  return 12;
}

function optionalColSpan(field) {
  if (field.type === "json") return 24;
  return 12;
}

function onField({ path, value }) {
  editor.patchFormField(path, value);
}

function optionalActive(field) {
  return optionalFieldActive(field, form.value);
}

function optionalToggleLabel(field) {
  if (field.type === "boolean") {
    return `Override — ${field.label}`;
  }
  return field.label;
}

function setOptionalActive(field, on) {
  editor.setForm(setOverrideEnabled(field, form.value, on));
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
  gap: 10px;
  margin-bottom: 4px;
  font-size: 14px;
}
</style>

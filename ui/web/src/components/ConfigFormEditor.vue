<template>
  <div class="config-form-editor">
    <el-alert
      v-if="parseError"
      type="warning"
      :title="parseError"
      show-icon
      :closable="false"
      class="mb-12"
    />
    <el-skeleton v-if="!schema" :rows="6" animated />

    <div v-else class="config-form-body">
      <el-tabs v-model="activeTab" class="config-tabs">
        <el-tab-pane v-for="tab in visibleTabs" :key="tab.id" :name="tab.id">
          <template #label>
            <span class="tab-label">{{ tab.label }}</span>
            <el-badge
              v-if="tabAttention(tab)"
              :value="tabAttention(tab)"
              type="warning"
              class="tab-badge"
            />
          </template>

          <p v-if="tab.description" class="tab-desc">{{ tab.description }}</p>

          <div class="tab-sections">
            <el-card
              v-for="sec in tab.sections"
              :key="sec.id"
              shadow="never"
              class="section-card"
            >
        <template #header>
          <div class="sec-header">
            <span class="section-title">{{ sec.title }}</span>
            <div v-if="attentionCount(sec)" class="sec-attention">
              <span
                v-if="unfilledRequiredCount(sec)"
                class="sec-attention-item"
                :title="`${unfilledRequiredCount(sec)} required field(s) empty`"
              >
                <span class="rf-label-required" aria-hidden="true">*</span>
                {{ unfilledRequiredCount(sec) }}
              </span>
            </div>
          </div>
        </template>

        <p v-if="sec.description" class="sec-desc">{{ sec.description }}</p>

        <template v-if="sec.id === 'preview'">
          <PreviewEntriesField
            :model-value="previewPromptsValue"
            :entry-fields="previewEntryFields"
            :parent-form="formValues"
            :capabilities="modelCapabilities"
            @update:model-value="onPreviewPromptsUpdate"
          />
          <div class="group-title">Global preview settings</div>
          <p class="sec-desc preview-global-hint">
            Defaults for all preview rows (schedule, size, seeds). Override per row in the dialog.
          </p>
        </template>

        <div v-if="adapterModeField(sec)" class="adapter-mode-row">
          <ConfigFormField
            :field="adapterModeField(sec)!"
            :form="formValues"
            :capabilities="modelCapabilities"
            @update:path="onFieldUpdate"
          />
        </div>

        <el-alert
          v-if="sec.id === 'model' && selectedCapability"
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
          v-if="sec.id === 'adapter' && selectedCapability"
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
          <template v-if="sectionFields(sec, 'required').length">
            <div class="group-title">Required</div>
            <el-row :gutter="16">
              <el-col
                v-for="field in sectionFields(sec, 'required')"
                :key="field.path"
                :xs="24"
                :sm="fieldColSpan(field)"
              >
                <ConfigFormField
                  :field="field"
                  :form="formValues"
                  :capabilities="modelCapabilities"
                  @update:path="onFieldUpdate"
                />
              </el-col>
            </el-row>
          </template>

          <template v-if="sectionFields(sec, 'recommended').length">
            <div class="group-title group-title--important">
              <template v-if="sec.id === 'preview'">Schedule &amp; toggles</template>
              <template v-else>
                Important
                <el-text type="info" size="small" class="group-hint">
                  Has a default in the trainer — review before running
                </el-text>
              </template>
            </div>
            <el-row :gutter="16">
              <el-col
                v-for="field in sectionFields(sec, 'recommended')"
                :key="field.path"
                :xs="24"
                :sm="fieldColSpan(field)"
              >
                <ConfigFormField
                  :field="field"
                  :form="formValues"
                  :capabilities="modelCapabilities"
                  @update:path="onFieldUpdate"
                />
              </el-col>
            </el-row>
          </template>

          <template v-if="sectionFields(sec, 'advanced').length">
            <div v-if="sec.id === 'preview'" class="group-title optional-title">
              Generation defaults
            </div>
            <el-row :gutter="16">
              <el-col
                v-for="field in sectionFields(sec, 'advanced')"
                :key="field.path"
                :xs="24"
                :sm="fieldColSpan(field)"
              >
                <ConfigFormField
                  :field="field"
                  :form="formValues"
                  :capabilities="modelCapabilities"
                  @update:path="onFieldUpdate"
                />
              </el-col>
            </el-row>
          </template>
        </el-form>
            </el-card>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import PreviewEntriesField from "./PreviewEntriesField.vue";
import type { PreviewEntry } from "../lib/previewEntries";
import {
  buildConfigFormTabs,
  configFieldColSpan,
  type ConfigSchemaSection,
} from "../lib/configFormSections";
import {
  fieldIsFilled,
  fieldVisible,
  getModelCapability,
  modelSupportsAdapters,
  trainingModesLabel,
} from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";
import type { FormValues, SchemaField } from "../types/forms";

interface ConfigFormTab {
  id: string;
  label: string;
  description?: string;
  sections: ConfigSchemaSection[];
}

const editor = useConfigEditorStore();
const { form, schema, parseError, modelCapabilities } = storeToRefs(editor);

const activeTab = ref("setup");

const formValues = computed(() => form.value ?? ({} as FormValues));

const selectedCapability = computed(() =>
  getModelCapability(modelCapabilities.value, formValues.value["model.type"])
);

const trainingModesText = computed(() => trainingModesLabel(selectedCapability.value));

const visibleTabs = computed(() => {
  if (!schema.value) return [];
  return buildConfigFormTabs(schema.value.sections as ConfigSchemaSection[] | undefined, sectionHasVisibleFields);
});

function tabAttention(tab: ConfigFormTab): number {
  let n = 0;
  for (const sec of tab.sections) {
    n += attentionCount(sec);
  }
  return n;
}

const sectionPartitions = computed(() => {
  const map = new Map<string, ReturnType<typeof partition>>();
  for (const tab of visibleTabs.value) {
    for (const sec of tab.sections) {
      map.set(sec.id, partition(sec));
    }
  }
  return map;
});

function sectionHasVisibleFields(sec: ConfigSchemaSection): boolean {
  if (adapterModeField(sec)) return true;
  if (sec.id === "preview") return true;
  const p = partition(sec);
  return p.required.length + p.recommended.length + p.advanced.length > 0;
}

function fieldImportance(field: SchemaField): "required" | "recommended" | "advanced" {
  if (field.path === "output_dir") return "advanced";
  if (field.importance === "required" || field.importance === "recommended" || field.importance === "advanced") {
    return field.importance;
  }
  if (field.required) return "required";
  if (field.recommended) return "recommended";
  return "advanced";
}

const ADAPTER_MODE_PATH = "_has_adapter";
/** Shown at top of config editor page, not in the Setup tab. */
const PINNED_TOP_FIELD_PATH = "run_name";
const PREVIEW_PROMPTS_PATH = "preview.prompts";

const previewEntryFields = computed(() => {
  const reg = schema.value?.registries as { preview_entry_fields?: SchemaField[] } | undefined;
  return reg?.preview_entry_fields ?? [];
});

const previewPromptsValue = computed(() => formValues.value[PREVIEW_PROMPTS_PATH]);

function onPreviewPromptsUpdate(entries: PreviewEntry[]): void {
  editor.patchFormField(PREVIEW_PROMPTS_PATH, entries.length ? entries : null);
}

function isPreviewListField(field: SchemaField): boolean {
  return field.path === PREVIEW_PROMPTS_PATH || field.type === "preview_entries";
}

function isPinnedAdapterField(sec: ConfigSchemaSection, field: SchemaField): boolean {
  return sec.id === "adapter" && field.path === ADAPTER_MODE_PATH;
}

function adapterModeField(sec: ConfigSchemaSection): SchemaField | null {
  if (sec.id !== "adapter") return null;
  const caps = modelCapabilities.value;
  const field = (sec.fields || []).find((f) => f.path === ADAPTER_MODE_PATH);
  if (!field || !fieldVisible(field, formValues.value, caps)) return null;
  return field;
}

function partition(sec: ConfigSchemaSection) {
  const caps = modelCapabilities.value;
  const values = formValues.value;
  const visible = (sec.fields || []).filter(
    (f) =>
      fieldVisible(f, values, caps) &&
      !isPinnedAdapterField(sec, f) &&
      f.path !== PINNED_TOP_FIELD_PATH &&
      !isPreviewListField(f)
  );
  const required = visible.filter((f) => fieldImportance(f) === "required");
  const recommended = visible.filter((f) => fieldImportance(f) === "recommended");
  const advanced = visible.filter((f) => fieldImportance(f) === "advanced");
  return { required, recommended, advanced };
}

function sectionFields(
  sec: ConfigSchemaSection,
  tier: "required" | "recommended" | "advanced"
): SchemaField[] {
  return sectionPartitions.value.get(sec.id)?.[tier] ?? [];
}

function unfilledRequiredCount(sec: ConfigSchemaSection): number {
  const p = sectionPartitions.value.get(sec.id);
  if (!p) return 0;
  const values = formValues.value;
  return p.required.filter((f) => !fieldIsFilled(f, values)).length;
}

function attentionCount(sec: ConfigSchemaSection): number {
  const requiredEmpty = unfilledRequiredCount(sec);
  let extra = 0;
  if (sec.id === "preview") {
    const prompts = formValues.value[PREVIEW_PROMPTS_PATH];
    if (!Array.isArray(prompts) || prompts.length === 0) {
      extra = 1;
    }
  }
  return requiredEmpty + extra;
}

function fieldColSpan(field: SchemaField): number {
  return configFieldColSpan(field);
}

function onFieldUpdate({ path, value }: { path: string; value: unknown }): void {
  editor.patchFormField(path, value);
}

onMounted(async () => {
  if (!schema.value) {
    await editor.fetchSchema();
  }
});

watch(
  visibleTabs,
  (tabs) => {
    if (!tabs.length) return;
    if (!tabs.some((t) => t.id === activeTab.value)) {
      activeTab.value = tabs[0].id;
    }
  },
  { immediate: true }
);
</script>

<style scoped>
.config-form-editor {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow-x: hidden;
}
.mb-12 {
  margin-bottom: 12px;
}
.config-form-body {
  width: 100%;
}
.config-tabs :deep(.el-tabs__content) {
  overflow: visible;
}
.tab-label {
  margin-right: 4px;
}
.tab-badge {
  margin-left: 4px;
  vertical-align: middle;
}
.tab-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
  overflow-wrap: anywhere;
}
.tab-sections {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
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

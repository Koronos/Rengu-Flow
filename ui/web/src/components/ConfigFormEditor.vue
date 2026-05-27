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
            <div v-if="attentionCount(sec)" class="sec-badges">
              <el-tag v-if="unfilledCount(sec).required" type="danger" size="small" effect="plain">
                {{ unfilledCount(sec).required }} required
              </el-tag>
              <el-tag
                v-if="unfilledCount(sec).recommended"
                type="warning"
                size="small"
                effect="plain"
              >
                {{ unfilledCount(sec).recommended }} important
              </el-tag>
            </div>
          </div>
        </template>

        <p v-if="sec.description" class="sec-desc">{{ sec.description }}</p>

        <div v-if="adapterModeField(sec)" class="adapter-mode-row">
          <ConfigFormField
            :field="adapterModeField(sec)"
            :form="form"
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
            Adapter types: <strong>{{ selectedCapability.adapters.join(", ") }}</strong>
            <span v-if="selectedCapability.full_finetune"> — or disable adapter for full finetune.</span>
          </template>
          <template v-else-if="selectedCapability.full_finetune">
            Full-model finetune only (no LoRA / LoKr).
          </template>
        </el-alert>

        <el-form label-position="top" class="config-form">
          <template v-if="partition(sec).required.length">
            <div class="group-title">Required</div>
            <el-row :gutter="16">
              <el-col
                v-for="field in partition(sec).required"
                :key="field.path"
                :xs="24"
                :sm="fieldColSpan(field)"
              >
                <ConfigFormField
                  :field="field"
                  :form="form"
                  :capabilities="modelCapabilities"
                  @update:path="onFieldUpdate"
                />
              </el-col>
            </el-row>
          </template>

          <template v-if="partition(sec).recommended.length">
            <div class="group-title group-title--important">
              Important
              <el-text type="info" size="small" class="group-hint">
                Has a default in the trainer — review before running
              </el-text>
            </div>
            <el-row :gutter="16">
              <el-col
                v-for="field in partition(sec).recommended"
                :key="field.path"
                :xs="24"
                :sm="fieldColSpan(field)"
              >
                <ConfigFormField
                  :field="field"
                  :form="form"
                  :capabilities="modelCapabilities"
                  @update:path="onFieldUpdate"
                />
              </el-col>
            </el-row>
          </template>

          <template v-if="partition(sec).advanced.length">
            <template v-if="sec.flat_optional">
              <div class="group-title optional-title">Optional</div>
              <el-row :gutter="16">
                <el-col
                  v-for="field in partition(sec).advanced"
                  :key="field.path"
                  :xs="24"
                  :sm="fieldColSpan(field)"
                >
                  <ConfigFormField
                    :field="field"
                    :form="form"
                    :capabilities="modelCapabilities"
                    @update:path="onFieldUpdate"
                  />
                </el-col>
              </el-row>
            </template>
            <el-collapse v-else v-model="advancedOpen[sec.id]" class="optional-collapse">
              <el-collapse-item :name="sec.id">
                <template #title>
                  <span class="group-title optional-title">
                    Advanced / optional ({{ partition(sec).advanced.length }})
                  </span>
                </template>
                <el-row :gutter="16">
                  <el-col
                    v-for="field in partition(sec).advanced"
                    :key="field.path"
                    :xs="24"
                    :sm="fieldColSpan(field)"
                  >
                    <ConfigFormField
                      :field="field"
                      :form="form"
                      :capabilities="modelCapabilities"
                      @update:path="onFieldUpdate"
                    />
                  </el-col>
                </el-row>
              </el-collapse-item>
            </el-collapse>
          </template>
        </el-form>
            </el-card>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormField from "./ConfigFormField.vue";
import { buildConfigFormTabs, configFieldColSpan } from "../lib/configFormSections";
import {
  fieldIsFilled,
  fieldVisible,
  getModelCapability,
  modelSupportsAdapters,
  trainingModesLabel,
} from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";

const editor = useConfigEditorStore();
const { form, schema, parseError, modelCapabilities, formVersion } = storeToRefs(editor);

const advancedOpen = reactive({});
const activeTab = ref("setup");

const selectedCapability = computed(() =>
  getModelCapability(modelCapabilities.value, form.value?.["model.type"])
);

const trainingModesText = computed(() => trainingModesLabel(selectedCapability.value));

const visibleTabs = computed(() => {
  if (!schema.value) return [];
  return buildConfigFormTabs(schema.value.sections, sectionHasVisibleFields);
});

function tabAttention(tab) {
  let n = 0;
  for (const sec of tab.sections) {
    n += attentionCount(sec);
  }
  return n;
}

function sectionHasVisibleFields(sec) {
  if (adapterModeField(sec)) return true;
  const p = partition(sec);
  return p.required.length + p.recommended.length + p.advanced.length > 0;
}

function fieldImportance(field) {
  if (field.path === "output_dir") return "advanced";
  if (field.importance) return field.importance;
  if (field.required) return "required";
  if (field.recommended) return "recommended";
  return "advanced";
}

const ADAPTER_MODE_PATH = "_has_adapter";

function isPinnedAdapterField(sec, field) {
  return sec.id === "adapter" && field.path === ADAPTER_MODE_PATH;
}

function adapterModeField(sec) {
  if (sec.id !== "adapter") return null;
  const caps = modelCapabilities.value;
  const field = (sec.fields || []).find((f) => f.path === ADAPTER_MODE_PATH);
  if (!field || !fieldVisible(field, form.value, caps)) return null;
  return field;
}

function partition(sec) {
  const caps = modelCapabilities.value;
  const values = form.value || {};
  const visible = (sec.fields || []).filter(
    (f) => fieldVisible(f, values, caps) && !isPinnedAdapterField(sec, f)
  );
  const required = visible.filter((f) => fieldImportance(f) === "required");
  const recommended = visible.filter((f) => fieldImportance(f) === "recommended");
  const advanced = visible.filter((f) => fieldImportance(f) === "advanced");
  return { required, recommended, advanced };
}

function unfilledCount(sec) {
  const p = partition(sec);
  const values = form.value || {};
  return {
    required: p.required.filter((f) => !fieldIsFilled(f, values)).length,
    recommended: p.recommended.filter((f) => !fieldIsFilled(f, values)).length,
  };
}

function attentionCount(sec) {
  const u = unfilledCount(sec);
  return u.required + u.recommended;
}

function fieldColSpan(field) {
  return configFieldColSpan(field);
}

function onFieldUpdate({ path, value }) {
  editor.patchFormField(path, value);
}

function initAdvancedOpen() {
  if (!schema.value) return;
  for (const sec of schema.value.sections || []) {
    if (partition(sec).advanced.length && !sec.flat_optional) {
      advancedOpen[sec.id] = [];
    }
  }
}

onMounted(async () => {
  if (!schema.value) {
    await editor.fetchSchema();
  }
  initAdvancedOpen();
});

watch(schema, () => {
  initAdvancedOpen();
});

watch(formVersion, () => {
  initAdvancedOpen();
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
.sec-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
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
  color: var(--el-color-warning);
}
.group-hint {
  margin-left: 8px;
  font-weight: 400;
}
.optional-title {
  font-weight: 500;
  color: var(--el-text-color-secondary);
}
.optional-collapse {
  border: none;
  margin-top: 4px;
}
.optional-collapse :deep(.el-collapse-item__header) {
  border: none;
  height: 36px;
  line-height: 36px;
}
.optional-collapse :deep(.el-collapse-item__wrap) {
  border: none;
}
</style>

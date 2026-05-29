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
            <ConfigFormSectionCard
              v-for="sec in tab.sections"
              :key="sec.id"
              :section="sec"
              :selected-capability="selectedCapability"
              :preview-entry-fields="previewEntryFields"
            />
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import ConfigFormSectionCard from "./ConfigFormSectionCard.vue";
import {
  buildConfigFormTabs,
  type ConfigFormTab,
  type ConfigSchemaSection,
} from "../lib/configFormSections";
import {
  sectionAttentionCount,
  sectionHasVisibleFields,
} from "../lib/configFormSectionLogic";
import { getModelCapability } from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";
import type { FormValues, SchemaField } from "../types/forms";

const editor = useConfigEditorStore();
const { form, schema, parseError, modelCapabilities } = storeToRefs(editor);

const activeTab = ref("setup");

const formValues = computed(() => form.value ?? ({} as FormValues));

const selectedCapability = computed(() =>
  getModelCapability(modelCapabilities.value, formValues.value["model.type"])
);

const visibleTabs = computed(() => {
  if (!schema.value) return [];
  return buildConfigFormTabs(
    schema.value.sections as ConfigSchemaSection[] | undefined,
    (sec) => sectionHasVisibleFields(sec, formValues.value, modelCapabilities.value)
  );
});

function tabAttention(tab: ConfigFormTab): number {
  return tab.sections.reduce(
    (n, sec) => n + sectionAttentionCount(sec, formValues.value, modelCapabilities.value),
    0
  );
}

const previewEntryFields = computed(() => {
  const reg = schema.value?.registries as { preview_entry_fields?: SchemaField[] } | undefined;
  return reg?.preview_entry_fields ?? [];
});

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
</style>

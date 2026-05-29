<template>
  <div class="dataset-augmentation-panel">
    <p class="tab-intro">
      Image diversity settings apply before VAE encode. Enable augmentation once for the dataset;
      customize individual <code>[[directory]]</code> rows with strategy overrides when needed.
      See
      <router-link :to="{ path: '/docs', query: { doc: 'docs/user/dataset-augmentation.md' } }">
        dataset augmentation
      </router-link>
      in Docs.
    </p>

    <el-alert v-if="catalogError" type="warning" :title="catalogError" show-icon class="mb-12" />

    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">Dataset-wide</span>
      </template>
      <p class="section-desc">
        Written to <code>[dataset.augmentation]</code> when enabled or when you set non-default
        options. Folders inherit these settings unless you customize a folder below.
      </p>
      <AugmentationConfigEditor
        :config="globalConfig"
        :catalog="catalog"
        :schema-fields="augmentationSchemaFields"
        show-advanced
        show-strategies
        @update="onGlobalUpdate"
      />
    </el-card>

    <el-card v-if="showPerFolderSection" shadow="never" class="section-card">
      <template #header>
        <span class="section-title">Per-directory overrides</span>
      </template>
      <p class="section-desc">
        Optional per <code>[[directory]]</code>. Rows without customization inherit the dataset-wide
        preset and strategy defaults above (same pattern as directory field overrides).
      </p>

      <el-empty
        v-if="!directories.length"
        description="No [[directory]] entries yet"
        :image-size="56"
      >
        <el-button type="primary" :icon="Plus" @click="emit('go-directories')">
          Add directory
        </el-button>
      </el-empty>

      <div v-else class="folder-list">
        <div v-for="(entry, index) in directories" :key="folderKey(entry, index)" class="folder-row">
          <div class="folder-head">
            <div class="folder-meta">
              <span class="folder-title">{{ folderTitle(entry, index) }}</span>
              <span class="folder-path">{{ entry.path || "path not set" }}</span>
            </div>
            <div v-if="hasDirectoryAugmentationOverride(entry)" class="folder-actions">
              <el-button type="primary" link @click="clearFolderCustomization(index)">
                Use dataset defaults
              </el-button>
            </div>
            <div v-else class="folder-actions">
              <el-button type="primary" link @click="startFolderCustomization(index)">
                Customize folder
              </el-button>
            </div>
          </div>

          <div v-if="!hasDirectoryAugmentationOverride(entry)" class="inherit-line">
            <el-text type="info" size="small">
              Inherits: {{ summarizeAugmentation(globalConfig, catalog) }}
            </el-text>
          </div>

          <AugmentationConfigEditor
            v-else
            :config="folderConfig(entry)"
            :catalog="catalog"
            :schema-fields="augmentationSchemaFields"
            :hide-enable="!folderNeedsFullEditor(entry)"
            :hide-preset="!folderNeedsFullEditor(entry)"
            :show-advanced="folderNeedsFullEditor(entry)"
            show-strategies
            @update="(cfg) => onFolderUpdate(index, cfg)"
          />
        </div>
      </div>
    </el-card>

    <el-text v-else type="info" size="small" class="per-folder-hint">
      Enable augmentation above to customize individual folders.
    </el-text>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { Plus } from "@element-plus/icons-vue";
import { storeToRefs } from "pinia";
import { api } from "../api";
import { basenameFromPath } from "../lib/datasetDirectoryForm";
import {
  compactStrategiesForStorage,
  directoryAugmentationNeedsFullEditor,
  emptyAugmentationConfig,
  hasDirectoryAugmentationOverride,
  isAugmentationEnabled,
  parseDirectoryAugmentation,
  parseGlobalAugmentation,
  serializeDirectoryAugmentation,
  serializeGlobalAugmentation,
  shouldWriteDirectoryAugmentation,
  shouldWriteGlobalAugmentation,
  summarizeAugmentation,
  type AugmentationCatalog,
  type AugmentationConfig,
} from "../lib/datasetAugmentation";
import { useDatasetEditorStore } from "../stores/datasetEditor";
import type { DirectoryFormRow } from "../lib/datasetDirectoryForm";
import type { SchemaField } from "../types/forms";
import AugmentationConfigEditor from "./AugmentationConfigEditor.vue";

const emit = defineEmits<{
  "go-directories": [];
}>();

const editor = useDatasetEditorStore();
const { form, schema } = storeToRefs(editor);

const augmentationSchemaFields = computed(
  () => (schema.value?.augmentation_directory_fields as SchemaField[] | undefined) ?? []
);

const catalog = ref<AugmentationCatalog | null>(null);
const catalogError = ref("");

const directories = computed<DirectoryFormRow[]>(() => {
  const dirs = form.value?._directories;
  return (Array.isArray(dirs) ? dirs : []) as DirectoryFormRow[];
});

const globalConfig = computed(() => {
  return parseGlobalAugmentation(form.value) ?? emptyAugmentationConfig();
});

const globalAugmentationOn = computed(() => isAugmentationEnabled(globalConfig.value));

const showPerFolderSection = computed(() => {
  if (globalAugmentationOn.value) return true;
  return directories.value.some((entry) => hasDirectoryAugmentationOverride(entry));
});

onMounted(async () => {
  try {
    catalog.value = (await api.getAugmentationCatalog()) as AugmentationCatalog;
  } catch (e) {
    catalogError.value = e instanceof Error ? e.message : "Could not load augmentation catalog";
  }
});

function folderConfig(entry: DirectoryFormRow): AugmentationConfig {
  const parsed = parseDirectoryAugmentation(entry);
  if (!parsed) return { ...globalConfig.value };
  return {
    ...globalConfig.value,
    ...parsed,
    strategies: parsed.strategies ? { ...parsed.strategies } : undefined,
  };
}

function folderNeedsFullEditor(entry: DirectoryFormRow): boolean {
  return directoryAugmentationNeedsFullEditor(parseDirectoryAugmentation(entry), globalConfig.value);
}

function folderKey(entry: DirectoryFormRow, index: number): string {
  return `aug-${index}:${(entry.path || "").trim()}`;
}

function folderTitle(entry: DirectoryFormRow, index: number): string {
  const path = (entry.path || "").trim();
  if (!path) return `Directory #${index + 1}`;
  return basenameFromPath(path) || path;
}

function onGlobalUpdate(config: AugmentationConfig) {
  if (!form.value) return;
  const stored = compactStrategiesForStorage(config, catalog.value);
  if (!shouldWriteGlobalAugmentation(stored)) {
    const next = { ...form.value };
    delete next._dataset_augmentation;
    editor.setForm(next);
    if (!stored.enabled) {
      clearAllFolderCustomizations();
    }
    return;
  }
  editor.patchFormField("_dataset_augmentation", serializeGlobalAugmentation(stored)!);
  if (!stored.enabled) {
    clearAllFolderCustomizations();
  }
}

function clearAllFolderCustomizations() {
  const next = directories.value.map((row) => {
    if (!hasDirectoryAugmentationOverride(row)) return row;
    const copy = { ...row };
    delete copy.augmentation;
    return copy;
  });
  if (next.some((row, i) => row !== directories.value[i])) {
    editor.patchDirectories(next);
  }
}

function startFolderCustomization(index: number) {
  const next = [...directories.value];
  const row = { ...next[index] };
  row.augmentation = { strategies: "{}" };
  next[index] = row;
  editor.patchDirectories(next);
}

function clearFolderCustomization(index: number) {
  const next = [...directories.value];
  const row = { ...next[index] };
  delete row.augmentation;
  next[index] = row;
  editor.patchDirectories(next);
}

function onFolderUpdate(index: number, config: AugmentationConfig) {
  const next = [...directories.value];
  const row = { ...next[index] };
  const stored = compactStrategiesForStorage(config, catalog.value);
  const serialized = serializeDirectoryAugmentation(stored, { global: globalConfig.value });
  if (!serialized) {
    delete row.augmentation;
  } else {
    row.augmentation = serialized;
  }
  next[index] = row;
  editor.patchDirectories(next);
}
</script>

<style scoped>
.dataset-augmentation-panel {
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
.per-folder-hint {
  display: block;
}
.mb-12 {
  margin-bottom: 12px;
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
.folder-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.folder-row {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  padding: 12px;
}
.folder-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.folder-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.folder-title {
  font-weight: 600;
}
.folder-path {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  word-break: break-all;
}
.inherit-line {
  margin-top: 8px;
}
</style>

<template>
  <el-dialog
    :model-value="visible"
    fullscreen
    :close-on-click-modal="false"
    :close-on-press-escape="!saving"
    :show-close="false"
    class="dataset-form-modal"
    @update:model-value="onToggle"
    @closed="onClosed"
  >
    <template #header>
      <div class="dfm__header">
        <span class="dfm__title">{{ editor.title }}</span>
        <div class="dfm__head-actions">
          <el-button :icon="CircleCheck" @click="onValidate">Validate</el-button>
          <el-button :icon="Upload" @click="triggerImport">Import TOML…</el-button>
          <el-button type="primary" :loading="saving" @click="onSave">Save</el-button>
          <el-divider direction="vertical" class="dfm__sep" />
          <el-tooltip content="Close" :show-after="300">
            <el-button :icon="Close" circle size="large" class="dfm__close" @click="modal.close()" />
          </el-tooltip>
        </div>
      </div>
    </template>

    <ImportTomlOverlay ref="importOverlay" @import="handleImport">
      <div class="dfm__body">
        <div class="title-row">
          <el-input
            v-model="datasetName"
            class="dataset-name-input"
            placeholder="Dataset name"
            maxlength="200"
            show-word-limit
          />
          <el-text v-if="!isNew && datasetId" type="info" size="small" class="dataset-id-hint">
            ID {{ datasetId }}
          </el-text>
        </div>

        <el-alert
          v-if="error"
          type="error"
          :title="error"
          show-icon
          class="mb-12"
          @close="editor.clearValidationErrorBar()"
        />
        <el-alert v-if="parseError" type="warning" :title="parseError" show-icon class="mb-12" />
        <el-alert
          v-if="message"
          type="success"
          :title="message"
          show-icon
          class="mb-12"
          @close="editor.clearValidationFeedback()"
        />

        <div v-loading="loading || (syncing && !form)" class="editor-body">
          <el-tabs v-model="formTab" class="dataset-form-tabs">
            <el-tab-pane label="Dataset defaults" name="global">
              <div v-if="form" :key="formVersion" class="dataset-form-tab-body">
                <DatasetFormGlobal />
              </div>
            </el-tab-pane>
            <el-tab-pane label="Directory" name="directories">
              <div v-if="form" :key="formVersion" class="dataset-form-tab-body">
                <DatasetFormFolders />
              </div>
            </el-tab-pane>
            <el-tab-pane label="Augmentation" name="augmentation">
              <div v-if="form" :key="formVersion" class="dataset-form-tab-body">
                <DatasetAugmentationPanel @go-directories="formTab = 'directories'" />
              </div>
            </el-tab-pane>
            <el-tab-pane label="TOML" name="toml">
              <div class="toml-pane-toolbar">
                <el-text type="info" size="small">Raw dataset TOML.</el-text>
                <el-button
                  :loading="exporting"
                  :disabled="isNew || !datasetId"
                  @click="exportDatasetToml"
                >
                  Export TOML
                </el-button>
              </div>
              <el-input
                v-model="tomlModel"
                type="textarea"
                :rows="24"
                class="toml-editor"
                placeholder="Dataset TOML…"
              />
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </ImportTomlOverlay>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { ElLoadingDirective, ElMessage } from "element-plus";
import { CircleCheck, Close, Upload } from "@element-plus/icons-vue";
import { api } from "../api";
import { downloadBlob } from "../lib/downloadBlob";
import { formatError } from "../lib/formatError";
import ImportTomlOverlay from "./ImportTomlOverlay.vue";
import DatasetFormGlobal from "./DatasetFormGlobal.vue";
import DatasetFormFolders from "./DatasetFormFolders.vue";
import DatasetAugmentationPanel from "./DatasetAugmentationPanel.vue";
import { useDatasetEditorStore } from "../stores/datasetEditor";
import { useDatasetFormModalStore } from "../stores/datasetFormModal";
import { formatDatasetLibraryRef } from "../lib/datasetLibraryRef";

const vLoading = ElLoadingDirective;
const editor = useDatasetEditorStore();
const modal = useDatasetFormModalStore();
const { visible, mode, editId, initialToml } = storeToRefs(modal);
const { datasetId, isNew, content, name, form, formVersion, loading, saving, syncing, error, message, parseError } =
  storeToRefs(editor);

const importOverlay = ref<InstanceType<typeof ImportTomlOverlay> | null>(null);
const formTab = ref("global");
const exporting = ref(false);

const datasetName = computed({
  get: () => name.value,
  set: (v: string) => editor.setName(v),
});
const tomlModel = computed({
  get: () => content.value,
  set: (v: string) => editor.setContent(v),
});

watch(visible, async (open) => {
  if (!open) return;
  formTab.value = "global";
  try {
    if (mode.value === "create") {
      await editor.openNew();
      if (initialToml.value) {
        await editor.applyToml(initialToml.value);
      }
    } else if (editId.value) {
      await editor.openExisting(editId.value);
    }
  } catch {
    /* store surfaces the error */
  }
});

function onValidate(): void {
  editor.validate().catch(() => {});
}

function triggerImport(): void {
  importOverlay.value?.openFilePicker?.();
}

async function handleImport(file: File): Promise<void> {
  try {
    const text = await file.text();
    await editor.applyToml(text);
    ElMessage.success("Imported TOML");
  } catch (e) {
    error.value = formatError(e);
  }
}

async function exportDatasetToml(): Promise<void> {
  if (isNew.value || !datasetId.value) return;
  exporting.value = true;
  try {
    const res = await api.exportDataset(datasetId.value);
    const text = res.content ?? "";
    const base = (name.value || `dataset_${datasetId.value}`).replace(/[^\w.-]+/g, "_");
    downloadBlob(new Blob([text], { type: "application/toml" }), `${base}.toml`);
    ElMessage.success("Exported TOML");
  } catch (e) {
    error.value = formatError(e);
    ElMessage.error(error.value);
  } finally {
    exporting.value = false;
  }
}

async function onSave(): Promise<void> {
  try {
    const { id } = await editor.save();
    modal.notifySaved({ id, name: name.value, ref: formatDatasetLibraryRef(id, name.value) });
    modal.close();
  } catch {
    /* store surfaces the error */
  }
}

function onToggle(value: boolean): void {
  if (!value) modal.close();
}

function onClosed(): void {
  editor.dispose();
}
</script>

<style scoped>
.dfm__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.dfm__title {
  font-size: 18px;
  font-weight: 600;
}
.dfm__head-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.dfm__sep {
  height: 1.6em;
  margin: 0 2px;
}
.dfm__close {
  font-size: 18px;
  color: var(--el-text-color-regular);
  border-color: var(--el-border-color);
}
.dfm__close:hover {
  color: var(--el-color-danger);
  border-color: var(--el-color-danger);
  background: var(--el-color-danger-light-9);
}
.title-row {
  display: flex;
  align-items: center;
  gap: var(--rf-space-sm);
  margin: 0 0 var(--rf-space-sm);
  flex-wrap: wrap;
}
.dataset-name-input {
  flex: 1;
  min-width: 200px;
  max-width: 480px;
}
.dataset-name-input :deep(.el-input__inner) {
  font-size: 20px;
  font-weight: 600;
}
.dataset-id-hint {
  flex-shrink: 0;
  font-family: ui-monospace, monospace;
}
.editor-body {
  margin-top: 4px;
}
.mb-12 {
  margin-bottom: 12px;
}
.toml-pane-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.toml-editor :deep(textarea) {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  line-height: 1.45;
}
</style>

<template>
  <div class="datasets-page page-shell">
    <div class="page-head editor-head">
      <el-button :icon="ArrowLeft" @click="router.push({ name: 'datasets-list' })">
        All datasets
      </el-button>
      <EditorActionBar variant="editor" class="editor-head-bar">
        <template #trailing>
          <EditorModeToggle v-model="editorMode" />
          <el-text v-if="editor.syncing" type="info" size="small" class="sync-hint">Syncing…</el-text>
        </template>
        <template #actions>
          <el-button :icon="CircleCheck" @click="onValidate">Validate</el-button>
          <el-button type="primary" :loading="editor.saving" @click="onSave">Save</el-button>
        </template>
      </EditorActionBar>
    </div>

    <div class="title-row">
      <el-input
        v-model="datasetName"
        class="dataset-name-input"
        placeholder="Dataset name"
        maxlength="200"
        show-word-limit
      />
      <el-text v-if="!editor.isNew && editor.datasetId" type="info" size="small" class="dataset-id-hint">
        ID {{ editor.datasetId }}
      </el-text>
    </div>

    <el-alert
      v-if="editor.error"
      type="error"
      :title="editor.error"
      show-icon
      class="mb-12"
      @close="editor.clearValidationErrorBar()"
    />
    <el-alert
      v-if="editor.parseError"
      type="warning"
      :title="editor.parseError"
      show-icon
      class="mb-12"
    />
    <el-alert
      v-if="editor.message"
      type="success"
      :title="editor.message"
      show-icon
      class="mb-12"
      @close="editor.clearValidationFeedback()"
    />

    <div
      v-show="editorMode === 'form'"
      v-loading="loading || (syncing && !form)"
      class="editor-body"
    >
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
      </el-tabs>
    </div>

    <div v-show="editorMode === 'toml'" class="editor-body">
      <div class="toml-pane-toolbar">
        <el-text type="info" size="small">
          Export uses absolute folder paths for use outside the UI.
        </el-text>
        <el-button :loading="exporting" :disabled="editor.isNew" @click="exportDatasetToml">
          Export TOML
        </el-button>
      </div>
      <el-input
        v-model="tomlModel"
        v-loading="editor.loading"
        type="textarea"
        :rows="tomlRows"
        class="toml-editor"
        placeholder="Dataset TOML…"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useRoute, useRouter } from "vue-router";
import { ElLoadingDirective, ElMessage } from "element-plus";
import { ArrowLeft, CircleCheck } from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";
import { downloadBlob } from "../lib/downloadBlob";
import { formatError } from "../lib/formatError";
import { useDatasetEditorStore } from "../stores/datasetEditor";
import EditorActionBar from "../components/EditorActionBar.vue";
import EditorModeToggle from "../components/EditorModeToggle.vue";
import DatasetFormGlobal from "../components/DatasetFormGlobal.vue";
import DatasetFormFolders from "../components/DatasetFormFolders.vue";
import DatasetAugmentationPanel from "../components/DatasetAugmentationPanel.vue";
import type { DatasetDetail } from "../types/api";

const route = useRoute();
const router = useRouter();
const { isMobile } = useBreakpoint();
const editor = useDatasetEditorStore();
const vLoading = ElLoadingDirective;
const { content, name, form, formVersion, loading, syncing } = storeToRefs(editor);

const editorMode = ref("form");
const formTab = ref("global");
const exporting = ref(false);

const datasetName = computed({
  get: () => name.value,
  set: (v) => editor.setName(v),
});

const tomlModel = computed({
  get: () => content.value,
  set: (value) => editor.setContent(value),
});

const tomlRows = computed(() => (isMobile.value ? 20 : 28));

async function load() {
  try {
    await editor.openFromRoute(route);
  } catch {
    router.replace({ name: "datasets-list" });
  }
}

function onSave() {
  editor.save(router).catch(() => {});
}

function onValidate() {
  editor.validate().catch(() => {});
}

async function exportDatasetToml() {
  const id = editor.datasetId;
  if (!id) return;
  exporting.value = true;
  try {
    const r = (await api.exportDataset(id)) as DatasetDetail & { filename?: string };
    const blob = new Blob([r.content ?? ""], { type: "application/toml;charset=utf-8" });
    downloadBlob(blob, r.filename || `dataset_${id}.toml`);
    ElMessage.success("Exported dataset TOML");
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    exporting.value = false;
  }
}

watch(() => route.fullPath, load, { immediate: true });

onUnmounted(() => {
  editor.dispose();
});
</script>

<style scoped>
.editor-head {
  margin-bottom: 0;
}
.editor-head-bar {
  flex: 1;
  min-width: 0;
}
.sync-hint {
  flex-shrink: 0;
}
.title-row {
  display: flex;
  align-items: center;
  gap: var(--rf-space-sm);
  margin: var(--rf-space-md) 0 var(--rf-space-sm);
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
.toml-pane-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.dataset-form-tabs {
  margin-top: 0;
}
.toml-editor :deep(textarea) {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  line-height: 1.45;
}
</style>

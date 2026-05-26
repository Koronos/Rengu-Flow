<template>
  <div>
    <h2 class="page-title">Dataset library</h2>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />
    <el-alert v-if="message" type="success" :title="message" show-icon class="mb-12" />

    <LibrarySelector
      ref="librarySelector"
      kind="dataset"
      :model-value="selected"
      :active-hint="selectedMeta"
      @update:model-value="onLibrarySelect"
      @open="open"
      @duplicate="duplicateById"
      @delete="removeById"
      @new-from-copy="newFromCopy"
    />

    <el-card shadow="never" class="mb-12">
      <el-space wrap>
        <el-button type="primary" :icon="Plus" @click="newDataset">New</el-button>
        <el-button :icon="Download" @click="importExample">Import example</el-button>
        <el-button :icon="Connection" @click="composeDialogVisible = true">Compose</el-button>
        <el-button type="success" :icon="Check" @click="save">Save</el-button>
        <el-button :icon="CircleCheck" @click="validate">Validate</el-button>
        <el-button :icon="View" @click="refreshPreview">Preview</el-button>
        <el-button v-if="selected" @click="exportToml">Export TOML</el-button>
      </el-space>
      <el-upload
        class="import-drop"
        drag
        :auto-upload="false"
        :show-file-list="false"
        accept=".toml"
        @change="onImportTomlFile"
      >
        <div class="import-drop-inner">Drop a dataset .toml to import into the library</div>
      </el-upload>
    </el-card>

    <el-row :gutter="12">
      <el-col :xs="24" :md="16">
        <el-card shadow="never">
          <template #header>
            <div class="editor-card-header">
              <span>{{ selected ? `Editing: ${selected}` : "New dataset" }}</span>
              <EditorModeToggle v-model="tab" />
            </div>
          </template>

          <DatasetFormEditor
            v-show="tab === 'form'"
            ref="formEditor"
            v-model="content"
            @preview="onPreview"
            @directory-select="previewDirectoryIndex = $event"
          />

          <el-input
            v-show="tab === 'toml'"
            v-model="content"
            type="textarea"
            :rows="isMobile ? 16 : 24"
            class="toml-editor"
            spellcheck="false"
            @blur="refreshPreview"
          />
        </el-card>
      </el-col>

      <el-col :xs="24" :md="8">
        <el-card shadow="never" class="preview-card">
          <template #header>Preview</template>
          <el-skeleton v-if="previewLoading" :rows="4" animated />
          <template v-else-if="preview">
            <p class="preview-summary">
              <strong>{{ preview.directory_count }}</strong> directories ·
              <strong>{{ preview.total_images }}</strong> images
              <span v-if="preview.total_videos"> · {{ preview.total_videos }} videos</span>
            </p>
            <div
              v-for="dir in preview.directories"
              :key="dir.index"
              class="dir-preview"
              :class="{ 'dir-preview--active': dir.index === previewDirectoryIndex }"
              @click="previewDirectoryIndex = dir.index"
            >
              <div class="dir-path">{{ dir.path || `(dir #${dir.index + 1})` }}</div>
              <el-tag v-if="dir.ok" type="success" size="small">
                {{ dir.image_count }} img
                <span v-if="dir.video_count"> + {{ dir.video_count }} vid</span>
              </el-tag>
              <el-tag v-else type="warning" size="small">{{ dir.error || "invalid" }}</el-tag>
              <span v-if="dir.num_repeats > 1" class="repeats">×{{ dir.num_repeats }}</span>
            </div>
            <p class="gallery-hint">Images for directory {{ previewDirectoryIndex + 1 }}</p>
            <DatasetImageGallery :content="content" :directory-index="previewDirectoryIndex" />
          </template>
          <el-empty v-else description="Save or Preview to scan folders" :image-size="48" />
        </el-card>
      </el-col>
    </el-row>

    <el-dialog v-model="newDialogVisible" title="New dataset id" width="90%" style="max-width: 400px">
      <el-input v-model="newDatasetId" placeholder="my_dataset" @keyup.enter="confirmNew" />
      <template #footer>
        <el-button @click="newDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="confirmNew">Create</el-button>
      </template>
    </el-dialog>

    <el-dialog
      v-model="composeDialogVisible"
      title="Compose datasets"
      width="90%"
      style="max-width: 520px"
    >
      <p class="compose-hint">
        Merge several library datasets into one TOML (all <code>[[directory]]</code> blocks combined).
      </p>
      <el-form label-position="top">
        <el-form-item label="New composed dataset id">
          <el-input v-model="composeTargetId" placeholder="combined_dataset" />
        </el-form-item>
        <el-form-item label="Source datasets (order preserved)">
          <el-select v-model="composeSourceIds" multiple filterable class="w-full">
            <el-option v-for="d in composeOptions" :key="d.id" :label="d.id" :value="d.id" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="composeDialogVisible = false">Cancel</el-button>
        <el-button type="primary" :disabled="!composeTargetId || !composeSourceIds.length" @click="doCompose">
          Compose
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, watch } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  Check,
  CircleCheck,
  Connection,
  CopyDocument,
  Delete,
  Download,
  Plus,
  View,
} from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";
import DatasetFormEditor from "../components/DatasetFormEditor.vue";
import EditorModeToggle from "../components/EditorModeToggle.vue";
import LibrarySelector from "../components/LibrarySelector.vue";
import DatasetImageGallery from "../components/DatasetImageGallery.vue";

const DEFAULT_TEMPLATE = `resolutions = [1024]
frame_buckets = [1]

[[directory]]
path = "/path/to/your/images"
num_repeats = 1
`;

const { isMobile } = useBreakpoint();

const librarySelector = ref(null);
const composeOptions = ref([]);
const selected = ref(null);
const selectedMeta = ref("");
const content = ref("");
const message = ref("");
const error = ref("");
const tab = ref("form");
const formEditor = ref(null);
const preview = ref(null);
const previewLoading = ref(false);
const newDialogVisible = ref(false);
const newDatasetId = ref("my_dataset");
const composeDialogVisible = ref(false);
const composeTargetId = ref("composed_dataset");
const composeSourceIds = ref([]);
const previewDirectoryIndex = ref(0);

async function loadComposeOptions() {
  const data = await api.searchDatasets({ q: "", page: 1, page_size: 100 });
  composeOptions.value = data.items || [];
}

watch(composeDialogVisible, (open) => {
  if (open) loadComposeOptions().catch(() => {});
});

function onPreview(p) {
  preview.value = p;
}

async function refreshPreview() {
  if (!content.value.trim()) {
    preview.value = null;
    return;
  }
  previewLoading.value = true;
  try {
    const r = await api.previewDataset(content.value);
    if (r.ok) preview.value = r.preview;
    else preview.value = null;
  } catch {
    preview.value = null;
  } finally {
    previewLoading.value = false;
  }
}

function onLibrarySelect(id) {
  selected.value = id || null;
}

async function open(id) {
  error.value = "";
  const data = await api.getDataset(id);
  selected.value = id;
  content.value = data.content;
  tab.value = "form";
  const row = composeOptions.value.find((d) => d.id === id);
  if (row?.directory_count != null) {
    selectedMeta.value = `${row.directory_count} directories`;
  } else {
    selectedMeta.value = "";
  }
  await formEditor.value?.reloadFromToml?.();
  await refreshPreview();
}

function newDataset() {
  selected.value = null;
  selectedMeta.value = "";
  content.value = DEFAULT_TEMPLATE;
  tab.value = "form";
  preview.value = null;
}

async function save() {
  error.value = "";
  message.value = "";
  try {
    if (selected.value) {
      await api.saveDataset(selected.value, content.value);
    } else {
      newDatasetId.value = "my_dataset";
      newDialogVisible.value = true;
      return;
    }
    message.value = "Saved.";
    ElMessage.success("Saved");
    librarySelector.value?.refreshBrowser?.();
    await refreshPreview();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function confirmNew() {
  const id = newDatasetId.value.trim();
  if (!id) return;
  newDialogVisible.value = false;
  try {
    await api.createDataset(id, content.value);
    selected.value = id;
    message.value = "Saved.";
    ElMessage.success("Created");
    librarySelector.value?.refreshBrowser?.();
    await refreshPreview();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function validate() {
  error.value = "";
  message.value = "";
  try {
    const r = await api.validateDataset(content.value);
    if (r.ok) {
      message.value = `Valid — ${r.preview?.total_images ?? "?"} images in ${r.preview?.directory_count ?? "?"} dirs`;
      ElMessage.success(message.value);
      if (r.preview) preview.value = r.preview;
    } else {
      error.value = r.error || "Invalid";
      ElMessage.error(error.value);
    }
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function duplicateById(id) {
  if (!id) return;
  try {
    const r = await api.duplicateDataset(id);
    message.value = `Duplicated as ${r.id}`;
    ElMessage.success(message.value);
    librarySelector.value?.refreshBrowser?.();
    await open(r.id);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function newFromCopy(id) {
  if (!id) return;
  try {
    const { value } = await ElMessageBox.prompt(
      "New dataset id (copy of " + id + ")",
      "New from copy",
      { inputValue: `${id}_copy`, inputPattern: /.+/ }
    );
    const src = await api.getDataset(id);
    await api.createDataset(value.trim(), src.content);
    ElMessage.success("Created");
    librarySelector.value?.refreshBrowser?.();
    await open(value.trim());
  } catch (e) {
    if (e !== "cancel") {
      error.value = String(e);
      ElMessage.error(String(e));
    }
  }
}

async function removeById(id) {
  if (!id) return;
  try {
    await ElMessageBox.confirm(`Delete dataset "${id}"?`, "Confirm", { type: "warning" });
    await api.deleteDataset(id);
    if (selected.value === id) {
      selected.value = null;
      selectedMeta.value = "";
      content.value = "";
      preview.value = null;
    }
    ElMessage.success("Deleted");
    librarySelector.value?.refreshBrowser?.();
  } catch (e) {
    if (e !== "cancel") {
      error.value = String(e);
      ElMessage.error(String(e));
    }
  }
}

async function importExample() {
  try {
    const r = await api.importDatasetExample("examples/minimal_dataset.toml");
    message.value = `Imported ${r.id}`;
    ElMessage.success(message.value);
    librarySelector.value?.refreshBrowser?.();
    await open(r.id);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function doCompose() {
  try {
    const r = await api.composeDatasets(composeTargetId.value.trim(), composeSourceIds.value);
    composeDialogVisible.value = false;
    message.value = `Composed as ${r.id}`;
    ElMessage.success(message.value);
    librarySelector.value?.refreshBrowser?.();
    await loadComposeOptions();
    await open(r.id);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "application/toml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function exportToml() {
  if (!selected.value) return;
  try {
    const r = await api.exportDataset(selected.value);
    downloadText(r.filename || `${selected.value}.toml`, r.content);
    ElMessage.success("Exported");
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function onImportTomlFile(uploadFile) {
  const file = uploadFile?.raw;
  if (!file) return;
  try {
    const text = await file.text();
    const base = file.name.replace(/\.toml$/i, "") || "imported_dataset";
    const r = await api.importDataset(text, base);
    librarySelector.value?.refreshBrowser?.();
    await open(r.id);
    ElMessage.success(`Imported as ${r.id}`);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

watch(
  () => content.value,
  () => {
    if (tab.value === "toml") return;
  }
);
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
.editor-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.preview-card {
  margin-bottom: 12px;
}
@media (min-width: 768px) {
  .preview-card {
    margin-bottom: 0;
  }
}
.preview-summary {
  margin: 0 0 12px;
  font-size: 14px;
}
.dir-preview {
  margin-bottom: 12px;
  padding: 8px;
  border-radius: var(--el-border-radius-base);
  border: 1px solid var(--el-border-color-lighter);
  cursor: pointer;
}
.dir-preview--active {
  border-color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
}
.gallery-hint {
  margin: 12px 0 4px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.dir-path {
  font-size: 12px;
  word-break: break-all;
  margin-bottom: 4px;
}
.samples {
  font-size: 11px;
  color: var(--el-text-color-secondary);
  margin-top: 4px;
}
.repeats {
  margin-left: 6px;
  font-size: 12px;
}
.compose-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.w-full {
  width: 100%;
}
.import-drop {
  margin-top: 12px;
  width: 100%;
}
.import-drop-inner {
  padding: 8px 0;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
</style>

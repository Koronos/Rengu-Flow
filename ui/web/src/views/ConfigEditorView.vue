<template>
  <ImportTomlOverlay ref="importOverlay" @import="onImportFile">
    <div class="page-shell">
      <div class="page-head editor-head">
        <el-button :icon="ArrowLeft" @click="goList">All configs</el-button>
        <div class="page-head-actions editor-head-actions">
          <EditorModeToggle v-model="editorTab" />
          <el-text v-if="syncing" type="info" size="small" class="sync-hint">Syncing…</el-text>
          <el-button
            v-if="pickForJob && configId"
            type="success"
            :icon="VideoPlay"
            @click="useConfigForJob"
          >
            Use for training job
          </el-button>
          <el-button type="primary" :icon="Check" :loading="saving" @click="save">
            Save
          </el-button>
          <el-button :icon="CircleCheck" @click="onValidate">Validate</el-button>
          <el-dropdown trigger="click" @command="onEditorCommand">
            <el-button>
              More
              <el-icon class="el-icon--right"><ArrowDown /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-item command="import">Import TOML…</el-dropdown-item>
              <el-dropdown-item command="import-example">Import example</el-dropdown-item>
              <el-dropdown-item v-if="configId" command="duplicate" divided>Duplicate</el-dropdown-item>
              <el-dropdown-item v-if="configId" command="delete">Delete</el-dropdown-item>
            </template>
          </el-dropdown>
        </div>
      </div>

      <el-text v-if="selectedMeta" type="info" size="small" class="meta-row">{{ selectedMeta }}</el-text>

      <el-alert
        v-if="validationErrors.length"
        type="error"
        show-icon
        class="mb-12 validation-alert"
        @close="editor.validationErrors = []"
      >
        <template #title>Configuration is not valid</template>
        <ul class="validation-errors">
          <li v-for="(msg, idx) in validationErrors" :key="idx">{{ msg }}</li>
        </ul>
      </el-alert>
      <el-alert v-else-if="error" type="error" :title="error" show-icon class="mb-12" />
      <el-alert v-if="message" type="success" :title="message" show-icon class="mb-12" />

      <el-alert
        v-if="continuation"
        type="info"
        :closable="false"
        show-icon
        class="mb-12 continue-banner"
      >
        <template #title>Continuing run</template>
        <p class="continue-text">
          Config loaded from the run folder TOML (<code>{{ continuation.run_dir }}</code>).
          Edit parameters (e.g. raise <code>epochs</code> or <code>max_steps</code>), then queue a job.
          Training resumes in the same folder and updates the snapshot TOML there.
        </p>
        <el-space wrap class="continue-actions">
          <el-button type="primary" @click="queueContinuation(false)">Add continuation to queue</el-button>
          <el-button type="success" @click="queueContinuation(true)">Start continuation now</el-button>
          <el-checkbox v-model="continuationSaveToLibrary">Also save as library config</el-checkbox>
          <el-input
            v-if="continuationSaveToLibrary"
            v-model="continuationLibraryId"
            placeholder="library config id"
            style="max-width: 220px"
          />
          <el-button link @click="onClearContinuation">Cancel</el-button>
        </el-space>
      </el-alert>

      <el-alert
        v-if="pickForJob"
        type="warning"
        :closable="false"
        show-icon
        class="mb-12 pick-banner"
      >
        <template #title>Choosing config for a training job</template>
        Validate if needed, then click <strong>Use for training job</strong>. You can also
        <el-button type="primary" link @click="cancelPick">return to Runs</el-button>
        without selecting.
      </el-alert>

      <div v-loading="loading || (syncing && !form)" class="editor-body">
        <div v-show="editorTab === 'form'" class="editor-pane">
          <ConfigFormEditor :key="formVersion" />
        </div>

        <div v-show="editorTab === 'toml'" class="editor-pane">
          <div class="toml-pane-toolbar">
            <el-text type="info" size="small">
              Export includes resolved dataset TOML files (ZIP) for CLI training.
            </el-text>
            <el-button :loading="exporting" @click="exportTrainingBundle">
              Export for CLI (ZIP)
            </el-button>
          </div>
          <el-input
            v-model="tomlModel"
            type="textarea"
            :rows="isMobile ? 16 : 24"
            class="toml-editor"
            spellcheck="false"
          />
        </div>
      </div>

      <el-dialog v-model="newDialogVisible" title="New config id" width="90%" style="max-width: 400px">
        <el-input v-model="newConfigId" placeholder="my_config" @keyup.enter="confirmNew" />
        <template #footer>
          <el-button @click="newDialogVisible = false">Cancel</el-button>
          <el-button type="primary" :loading="saving" @click="confirmNew">Create</el-button>
        </template>
      </el-dialog>
    </div>
  </ImportTomlOverlay>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  ArrowDown,
  ArrowLeft,
  Check,
  CircleCheck,
  VideoPlay,
} from "@element-plus/icons-vue";
import { api } from "../api";
import { downloadBlob } from "../lib/downloadBlob.js";
import { formatError } from "../lib/formatError";
import { useBreakpoint } from "../composables/useBreakpoint";
import ConfigFormEditor from "../components/ConfigFormEditor.vue";
import EditorModeToggle from "../components/EditorModeToggle.vue";
import ImportTomlOverlay from "../components/ImportTomlOverlay.vue";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";
import { useConfigEditorStore } from "../stores/configEditor";

const { isMobile } = useBreakpoint();
const route = useRoute();
const router = useRouter();
const editor = useConfigEditorStore();

const {
  configId,
  selectedMeta,
  content,
  form,
  loading,
  saving,
  syncing,
  error,
  message,
  validationErrors,
  formVersion,
  editorTab,
  continuation,
  continuationSaveToLibrary,
  continuationLibraryId,
} = storeToRefs(editor);

const pickForJob = computed(() => route.query.pick === "job");
const importOverlay = ref(null);
const newDialogVisible = ref(false);
const newConfigId = ref("my_config");
const exporting = ref(false);

const tomlModel = computed({
  get: () => content.value,
  set: (value) => editor.setContent(value),
});

function goList() {
  const query = pickForJob.value ? { pick: "job" } : {};
  router.push({ name: "configs-list", query });
}

async function load() {
  try {
    await editor.openFromRoute(route);
  } catch {
    goList();
  }
}

onUnmounted(() => {
  editor.dispose();
});

watch(() => route.fullPath, load, { immediate: true });

async function save() {
  if (configId.value) {
    await editor.saveExisting();
    return;
  }
  newConfigId.value = "my_config";
  newDialogVisible.value = true;
}

async function confirmNew() {
  const id = newConfigId.value.trim();
  if (!id) return;
  newDialogVisible.value = false;
  try {
    const ok = await editor.createNew(id);
    if (ok) {
      await router.replace({
        name: "configs-detail",
        params: { configId: id },
        query: pickForJob.value ? { pick: "job" } : {},
      });
    }
  } catch {
    /* store shows error */
  }
}

function onValidate() {
  return editor.validateConfig();
}

function onClearContinuation() {
  editor.clearContinuation();
  router.replace({ name: "configs-list" });
}

async function queueContinuation(startNow) {
  try {
    const job = await editor.queueContinuation({
      startNow,
      saveToLibrary: continuationSaveToLibrary.value,
      libraryId: continuationLibraryId.value,
    });
    if (job) router.push({ name: "job-detail", params: { id: job.id } });
  } catch {
    /* store shows error */
  }
}

function cancelPick() {
  router.push({ name: "jobs" });
}

async function useConfigForJob() {
  if (!configId.value) {
    ElMessage.warning("Save the config first");
    return;
  }
  try {
    const r = await editor.validateConfig({ quiet: true });
    if (!r.ok) {
      await ElMessageBox.confirm(
        `${r.errors?.[0] || "Validation failed"}. Use this config for training anyway?`,
        "Config not valid",
        { type: "warning", confirmButtonText: "Use anyway", cancelButtonText: "Keep editing" }
      );
    }
  } catch (e) {
    if (e === "cancel") return;
    return;
  }
  setJobConfigId(configId.value);
  ElMessage.success(`"${configId.value}" selected for training`);
  router.push({ name: "jobs" });
}

async function duplicateCurrent() {
  if (!configId.value) return;
  try {
    const r = await api.duplicate(configId.value);
    ElMessage.success(`Duplicated as ${r.id}`);
    await router.push({ name: "configs-detail", params: { configId: String(r.id) } });
  } catch (e) {
    editor.error = formatError(e);
    ElMessage.error(editor.error);
  }
}

async function removeCurrent() {
  if (!configId.value) return;
  try {
    await ElMessageBox.confirm(`Delete config "${configId.value}"?`, "Confirm", { type: "warning" });
    await api.deleteConfig(configId.value);
    ElMessage.success("Deleted");
    goList();
  } catch (e) {
    if (e !== "cancel") {
      editor.error = formatError(e);
      ElMessage.error(editor.error);
    }
  }
}

async function importExample() {
  try {
    const r = await api.importExample("examples/minimal_config_lora_sdxl.toml");
    editor.message = `Imported ${r.id}`;
    ElMessage.success(editor.message);
    await router.push({ name: "configs-detail", params: { configId: String(r.id) } });
  } catch (e) {
    editor.error = formatError(e);
    ElMessage.error(editor.error);
  }
}

async function exportTrainingBundle() {
  const text = (content.value || "").trim();
  if (!text) {
    ElMessage.warning("Nothing to export — add training config TOML first.");
    return;
  }
  exporting.value = true;
  try {
    const name = configId.value ? String(configId.value) : "training_export";
    const { blob, filename } = await api.exportConfigBundle(name, text);
    downloadBlob(blob, filename);
    ElMessage.success("Exported ZIP for CLI");
  } catch (e) {
    editor.error = formatError(e);
    ElMessage.error(editor.error);
  } finally {
    exporting.value = false;
  }
}

function onEditorCommand(cmd) {
  if (cmd === "import") triggerImport();
  else if (cmd === "import-example") importExample();
  else if (cmd === "duplicate") duplicateCurrent();
  else if (cmd === "delete") removeCurrent();
}

function triggerImport() {
  importOverlay.value?.openFilePicker?.();
}

async function onImportFile(file) {
  try {
    const text = await file.text();
    const base = file.name.replace(/\.toml$/i, "") || "imported";
    const r = await api.importConfig(text, base);
    await router.push({ name: "configs-detail", params: { configId: String(r.id) } });
    ElMessage.success(`Imported as ${r.id}`);
  } catch (e) {
    editor.error = formatError(e);
    ElMessage.error(editor.error);
  }
}
</script>

<style scoped>
.editor-head {
  margin-bottom: 0;
}
.editor-head-actions {
  gap: var(--rf-space-xs);
}
.sync-hint {
  flex-shrink: 0;
}
.meta-row {
  display: block;
  margin: 0 0 var(--rf-space-sm);
}
.pick-banner,
.continue-banner {
  line-height: 1.5;
}
.continue-banner :deep(.el-alert__content) {
  width: 100%;
}
.continue-text {
  margin: 8px 0 12px;
  font-size: 13px;
  line-height: 1.5;
}
.continue-actions {
  margin-top: 4px;
}
.validation-alert :deep(.el-alert__content) {
  width: 100%;
}
.validation-errors {
  margin: 8px 0 0;
  padding-left: 1.25rem;
  line-height: 1.5;
}
.validation-errors li {
  margin-bottom: 4px;
}
.editor-pane {
  width: 100%;
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
.toml-editor {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
}
.toml-editor :deep(textarea) {
  font-family: inherit;
  font-size: inherit;
}
</style>

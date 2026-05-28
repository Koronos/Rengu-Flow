<template>
  <ImportTomlOverlay ref="importOverlay" @import="importConfigFile">
    <div class="page-shell">
      <div class="page-head editor-head">
        <el-button :icon="ArrowLeft" @click="goList">All configs</el-button>
        <EditorActionBar variant="editor" class="editor-head-bar">
          <template #trailing>
            <EditorModeToggle v-model="editorTab" />
            <el-text v-if="syncing" type="info" size="small" class="sync-hint">Syncing…</el-text>
          </template>
          <template #actions>
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
            <el-button @click="triggerImport">Import TOML…</el-button>
          </template>
        </EditorActionBar>
      </div>

      <div class="title-row">
        <div class="run-name-field">
          <span class="run-name-label">
            Run name
            <FieldHelpIcon v-if="runNameField" :field="runNameField" />
          </span>
          <el-input
            v-model="runNameModel"
            class="config-run-name-input"
            placeholder="Optional — timestamp-only folder if empty"
            maxlength="120"
            show-word-limit
          />
        </div>
        <el-text v-if="configId" type="info" size="small" class="config-id-hint">
          Config #{{ configId }}
        </el-text>
      </div>
      <el-text v-if="selectedMeta" type="info" size="small" class="config-meta-hint">
        {{ selectedMeta }}
      </el-text>

      <el-alert
        v-if="validationErrors.length"
        type="error"
        show-icon
        class="mb-12 validation-alert"
        @close="editor.clearValidationFeedback()"
      >
        <template #title>Configuration is not valid</template>
        <ul class="validation-errors">
          <li v-for="(msg, idx) in validationErrors" :key="idx">{{ msg }}</li>
        </ul>
      </el-alert>
      <el-alert
        v-else-if="error"
        type="error"
        :title="error"
        show-icon
        class="mb-12"
        @close="editor.clearValidationErrorBar()"
      />
      <el-alert
        v-if="message"
        type="success"
        :title="message"
        show-icon
        class="mb-12"
        @close="editor.clearValidationFeedback()"
      />

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

      <PickForJobBanner v-if="pickForJob" variant="editor" @cancel="cancelPick" />

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
          />
        </div>
      </div>

      <el-dialog v-model="newDialogVisible" title="New config id" width="90%" style="max-width: 400px">
        <el-input v-model="newConfigId" placeholder="my_config" />
        <template #footer>
          <el-button @click="newDialogVisible = false">Cancel</el-button>
          <el-button type="primary" :loading="saving" @click="confirmNew">Create</el-button>
        </template>
      </el-dialog>
    </div>
  </ImportTomlOverlay>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import { ArrowLeft, Check, CircleCheck, VideoPlay } from "@element-plus/icons-vue";
import { api } from "../api";
import { downloadBlob } from "../lib/downloadBlob";
import { formatError } from "../lib/formatError";
import { useBreakpoint } from "../composables/useBreakpoint";
import ConfigFormEditor from "../components/ConfigFormEditor.vue";
import EditorActionBar from "../components/EditorActionBar.vue";
import EditorModeToggle from "../components/EditorModeToggle.vue";
import ImportTomlOverlay from "../components/ImportTomlOverlay.vue";
import FieldHelpIcon from "../components/FieldHelpIcon.vue";
import PickForJobBanner from "../components/PickForJobBanner.vue";
import { useImportConfigToml } from "../composables/useImportConfigToml";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";
import { useConfigEditorStore } from "../stores/configEditor";
import type { JobRecord } from "../types/api";
import type { SchemaField } from "../types/forms";
import type ImportTomlOverlayType from "../components/ImportTomlOverlay.vue";

const { isMobile } = useBreakpoint();
const route = useRoute();
const router = useRouter();
const editor = useConfigEditorStore();
const vLoading = ElLoadingDirective;

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
  schema,
} = storeToRefs(editor);

const pickForJob = computed(() => route.query.pick === "job");
const importOverlay = ref<InstanceType<typeof ImportTomlOverlayType> | null>(null);
const { importConfigFile } = useImportConfigToml({ onError: (msg) => { editor.error = msg; } });
const newDialogVisible = ref(false);
const newConfigId = ref("my_config");
const exporting = ref(false);

const tomlModel = computed({
  get: () => content.value,
  set: (value) => editor.setContent(value),
});

const runNameModel = computed({
  get: () => (typeof form.value?.run_name === "string" ? form.value.run_name : ""),
  set: (value: string) => editor.patchFormField("run_name", value.trim()),
});

const runNameField = computed<SchemaField | null>(() => {
  const sections = (schema.value?.sections as { fields?: SchemaField[] }[] | undefined) ?? [];
  for (const sec of sections) {
    const field = sec.fields?.find((f) => f.path === "run_name");
    if (field) return field;
  }
  return null;
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

async function queueContinuation(startNow: boolean) {
  try {
    const job = (await editor.queueContinuation({
      startNow,
      saveToLibrary: continuationSaveToLibrary.value,
      libraryId: continuationLibraryId.value,
    })) as (JobRecord & { id?: string }) | null;
    if (job?.id) router.push({ name: "job-detail", params: { id: String(job.id) } });
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

function triggerImport() {
  importOverlay.value?.openFilePicker?.();
}
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
  align-items: flex-end;
  gap: var(--rf-space-sm);
  margin: var(--rf-space-md) 0 var(--rf-space-xs);
  flex-wrap: wrap;
}
.run-name-field {
  flex: 1;
  min-width: 200px;
  max-width: 480px;
}
.run-name-label {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 500;
  color: var(--el-text-color-regular);
}
.config-run-name-input {
  width: 100%;
}
.config-run-name-input :deep(.el-input__inner) {
  font-size: 20px;
  font-weight: 600;
}
.config-id-hint {
  flex-shrink: 0;
  font-family: var(--rf-font-mono, ui-monospace, monospace);
}
.config-meta-hint {
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

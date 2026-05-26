<template>
  <div>
    <h2 class="page-title">Configuration library</h2>

    <el-alert
      v-if="validationErrors.length"
      type="error"
      show-icon
      class="mb-12 validation-alert"
      @close="validationErrors = []"
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
        <el-button link @click="clearContinuation">Cancel</el-button>
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
      Open a config below, edit if needed, validate, then click
      <strong>Use for training job</strong>. You can also
      <el-button type="primary" link @click="cancelPick">return to Jobs</el-button>
      without selecting.
    </el-alert>

    <LibrarySelector
      ref="librarySelector"
      kind="config"
      :model-value="selected"
      :pick-for-job="pickForJob"
      :active-hint="selectedMeta"
      @update:model-value="onLibrarySelect"
      @open="open"
      @duplicate="duplicateById"
      @delete="removeById"
      @start-job="startJobFromConfig"
      @use-for-job="useConfigForJobById"
      @new-from-copy="newFromCopy"
    />

    <el-card shadow="never" class="mb-12">
      <el-space wrap>
        <el-button
          v-if="pickForJob && selected"
          type="success"
          :icon="VideoPlay"
          @click="useConfigForJob"
        >
          Use for training job
        </el-button>
        <el-button type="primary" :icon="Plus" @click="newConfig">New</el-button>
        <el-button :icon="Download" @click="importExample">Import example</el-button>
        <el-button v-if="selected" @click="exportToml">Export TOML</el-button>
        <el-button type="success" :icon="Check" @click="save">Save</el-button>
        <el-button :icon="CircleCheck" @click="validate">Validate</el-button>
      </el-space>
      <el-upload
        class="import-drop"
        drag
        :auto-upload="false"
        :show-file-list="false"
        accept=".toml"
        @change="onImportTomlFile"
      >
        <div class="import-drop-inner">Drop a .toml file to import into the library</div>
      </el-upload>
    </el-card>

    <el-card shadow="never">
          <template #header>
            <div class="editor-card-header">
              <span>{{ selected ? `Editing: ${selected}` : "New config" }}</span>
              <EditorModeToggle v-model="tab" />
            </div>
          </template>

          <ConfigFormEditor
            v-show="tab === 'form'"
            ref="formEditor"
            v-model="content"
          />

          <el-input
            v-show="tab === 'toml'"
            v-model="content"
            type="textarea"
            :rows="isMobile ? 16 : 24"
            class="toml-editor"
            spellcheck="false"
          />
    </el-card>

    <el-dialog v-model="newDialogVisible" title="New config id" width="90%" style="max-width: 400px">
      <el-input v-model="newConfigId" placeholder="my_config" @keyup.enter="confirmNew" />
      <template #footer>
        <el-button @click="newDialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="confirmNew">Create</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import {
  Check,
  CircleCheck,
  CopyDocument,
  Delete,
  Download,
  Plus,
  VideoPlay,
} from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";
import ConfigFormEditor from "../components/ConfigFormEditor.vue";
import EditorModeToggle from "../components/EditorModeToggle.vue";
import LibrarySelector from "../components/LibrarySelector.vue";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";

const DEFAULT_TEMPLATE = `dataset = "examples/minimal_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"

[adapter]
type = "lora"
rank = 16

[optimizer]
type = "adamw"
lr = 1.0e-4

lr_scheduler = "cosine"
[lr_scheduler_args]
lr_min = 0.0

epochs = 1
gradient_accumulation_steps = 1
micro_batch_size_per_gpu = 1
synthetic_num_batches = 50
logging_steps = 1
save_every_n_epochs = 1
output_dir = "output"
`;

const { isMobile } = useBreakpoint();
const route = useRoute();
const router = useRouter();

const pickForJob = computed(() => route.query.pick === "job");

const librarySelector = ref(null);
const selected = ref(null);
const selectedMeta = ref("");
const content = ref("");
const message = ref("");
const error = ref("");
const validationErrors = ref([]);
const tab = ref("form");
const formEditor = ref(null);
const newDialogVisible = ref(false);
const newConfigId = ref("my_config");
const continuation = ref(null);
const continuationSaveToLibrary = ref(false);
const continuationLibraryId = ref("");

async function applyRouteSelection() {
  const q = route.query.config;
  if (typeof q === "string" && q) {
    try {
      await open(q);
      return;
    } catch {
      /* config may have been deleted */
    }
  }
  if (pickForJob.value) {
    const stored = getJobConfigId();
    if (stored) {
      try {
        await open(stored);
      } catch {
        /* ignore */
      }
    }
  }
}

async function loadContinuation(runPath) {
  const data = await api.getRunConfig(runPath);
  continuation.value = {
    run_dir: data.run_dir,
    resume_from: data.resume_from,
  };
  continuationLibraryId.value = `${data.run_dir.split("/").pop()}_continued`;
  selected.value = null;
  selectedMeta.value = "from run folder";
  validationErrors.value = [];
  content.value = data.content;
  tab.value = "form";
  await formEditor.value?.reloadFromToml?.();
}

function clearContinuation() {
  continuation.value = null;
  router.replace({ name: "configs", query: {} });
}

async function queueContinuation(startNow) {
  if (!continuation.value) return;
  error.value = "";
  try {
    if (tab.value === "form" && formEditor.value?.flushToml) {
      await formEditor.value.flushToml();
    }
    const job = await api.continueRun({
      run_path: continuation.value.run_dir,
      content: content.value,
      save_to_library: continuationSaveToLibrary.value,
      config_id: continuationSaveToLibrary.value
        ? continuationLibraryId.value.trim() || undefined
        : undefined,
      enqueue: !startNow,
      start_immediately: startNow,
    });
    ElMessage.success(startNow ? "Continuation started" : "Continuation queued");
    clearContinuation();
    router.push(`/jobs/${job.id}`);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

onMounted(async () => {
  try {
    const cr = route.query.continue_run;
    if (typeof cr === "string" && cr) {
      await loadContinuation(cr);
    } else if (route.query.new === "1") {
      newConfig();
    } else {
      await applyRouteSelection();
    }
  } catch (e) {
    error.value = String(e);
  }
});

watch(
  () => route.query.config,
  () => {
    applyRouteSelection().catch((e) => {
      error.value = String(e);
    });
  }
);

watch(
  () => route.query.continue_run,
  (cr) => {
    if (typeof cr === "string" && cr) {
      loadContinuation(cr).catch((e) => {
        error.value = String(e);
      });
    }
  }
);

function cancelPick() {
  router.push({ name: "jobs" });
}

async function useConfigForJob() {
  if (!selected.value) {
    ElMessage.warning("Search and open a config first");
    return;
  }
  try {
    const r = await api.validate(content.value);
    if (!r.ok) {
      await ElMessageBox.confirm(
        `${r.error || "Validation failed"}. Use this config for training anyway?`,
        "Config not valid",
        { type: "warning", confirmButtonText: "Use anyway", cancelButtonText: "Keep editing" }
      );
    }
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
    return;
  }
  setJobConfigId(selected.value);
  ElMessage.success(`"${selected.value}" selected for training`);
  router.push({ name: "jobs" });
}

function onLibrarySelect(id) {
  selected.value = id || null;
}

async function open(id) {
  error.value = "";
  validationErrors.value = [];
  const data = await api.getConfig(id);
  selected.value = id;
  content.value = data.content;
  tab.value = "form";
  try {
    const summary = await api.searchConfigs({ q: id, page: 1, page_size: 1 });
    const row = summary.items?.find((c) => c.id === id);
    if (row) {
      const parts = [];
      if (row.model_type) parts.push(row.model_type);
      if (row.dataset_ref) parts.push(row.dataset_ref);
      selectedMeta.value = parts.join(" · ");
    } else {
      selectedMeta.value = "";
    }
  } catch {
    selectedMeta.value = "";
  }
  await formEditor.value?.reloadFromToml?.();
}

function newConfig() {
  selected.value = null;
  selectedMeta.value = "";
  validationErrors.value = [];
  content.value = DEFAULT_TEMPLATE;
  tab.value = "form";
}

async function save() {
  error.value = "";
  message.value = "";
  try {
    if (selected.value) {
      await api.saveConfig(selected.value, content.value);
    } else {
      newConfigId.value = "my_config";
      newDialogVisible.value = true;
      return;
    }
    message.value = "Saved.";
    ElMessage.success("Saved");
    librarySelector.value?.refreshBrowser?.();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function confirmNew() {
  const id = newConfigId.value.trim();
  if (!id) return;
  newDialogVisible.value = false;
  try {
    await api.createConfig(id, content.value);
    selected.value = id;
    message.value = "Saved.";
    ElMessage.success("Created");
    librarySelector.value?.refreshBrowser?.();
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function validate() {
  error.value = "";
  validationErrors.value = [];
  message.value = "";
  try {
    if (tab.value === "form" && formEditor.value?.flushToml) {
      await formEditor.value.flushToml();
    }
    const r = await api.validate(content.value);
    if (r.ok) {
      const res = r.resolution || {};
      const parts = [];
      if (res.optimizer?.available) {
        parts.push(`optimizer → ${res.optimizer.resolved_class || res.optimizer.name}`);
      }
      if (res.scheduler?.available) {
        parts.push(`scheduler → ${res.scheduler.resolved || res.scheduler.resolved_class || res.scheduler.name}`);
      }
      message.value = parts.length ? `Valid (${parts.join("; ")})` : "Valid.";
      ElMessage.success(message.value);
    } else {
      validationErrors.value =
        Array.isArray(r.errors) && r.errors.length ? r.errors : [r.error || "Invalid configuration."];
      ElMessage.error(
        validationErrors.value.length === 1
          ? validationErrors.value[0]
          : `${validationErrors.value.length} issues — see list above`
      );
    }
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}

async function duplicateById(id) {
  if (!id) return;
  try {
    const r = await api.duplicate(id);
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
      "New config id (copy of " + id + ")",
      "New from copy",
      { inputValue: `${id}_copy`, inputPattern: /.+/ }
    );
    const src = await api.getConfig(id);
    await api.createConfig(value.trim(), src.content);
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
    await ElMessageBox.confirm(`Delete config "${id}"?`, "Confirm", { type: "warning" });
    await api.deleteConfig(id);
    if (selected.value === id) {
      selected.value = null;
      selectedMeta.value = "";
      content.value = "";
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

async function startJobFromConfig(id) {
  if (!id) return;
  setJobConfigId(id);
  router.push({ name: "jobs" });
  ElMessage.success(`"${id}" selected — configure GPUs on Jobs`);
}

async function useConfigForJobById(id) {
  if (!id) return;
  if (id !== selected.value) await open(id);
  await useConfigForJob();
}

async function importExample() {
  try {
    const r = await api.importExample("examples/minimal_config_lora_sdxl.toml");
    message.value = `Imported ${r.id}`;
    ElMessage.success(message.value);
    librarySelector.value?.refreshBrowser?.();
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
    const r = await api.exportConfig(selected.value);
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
    const base = file.name.replace(/\.toml$/i, "") || "imported";
    const r = await api.importConfig(text, base);
    librarySelector.value?.refreshBrowser?.();
    await open(r.id);
    ElMessage.success(`Imported as ${r.id}`);
  } catch (e) {
    error.value = String(e);
    ElMessage.error(String(e));
  }
}
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
.pick-banner {
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

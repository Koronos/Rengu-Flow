<template>
  <el-dialog
    :model-value="modelValue"
    fullscreen
    :show-close="true"
    :close-on-click-modal="false"
    :close-on-press-escape="!submitting"
    class="run-form-modal"
    @update:model-value="onDialogToggle"
    @closed="onClosed"
  >
    <template #header>
      <div class="run-form-modal__header">
        <span class="run-form-modal__title">{{ title }}</span>
        <div class="run-form-modal__header-actions">
          <el-button :icon="CircleCheck" @click="onValidate">Validate</el-button>
          <el-button :icon="Upload" @click="triggerImport">Import TOML…</el-button>
        </div>
      </div>
    </template>

    <ImportTomlOverlay ref="importOverlay" @import="handleImport">
      <div v-loading="loading || (syncing && !form)" class="run-form-modal__body">
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
        <el-alert v-if="message" type="success" :title="message" show-icon class="mb-12" />

        <el-alert
          v-if="continuation"
          type="info"
          :closable="false"
          show-icon
          class="mb-12"
        >
          <template #title>Continuing run</template>
          <span class="continue-text">
            Resumes the run folder <code>{{ continuation.run_dir }}</code>. Pick a checkpoint in the
            Queue tab (or start from scratch). Raise <code>epochs</code>/<code>max_steps</code> to train further.
          </span>
        </el-alert>

        <el-tabs v-model="activeTab" class="run-form-tabs">
          <el-tab-pane label="Setup" name="setup">
            <div class="tab-sections">
              <ConfigFormSectionCard
                v-for="sec in setupSections"
                :key="sec.id"
                :section="sec"
                :selected-capability="selectedCapability"
                :preview-entry-fields="previewEntryFields"
              />
            </div>
          </el-tab-pane>

          <el-tab-pane label="Datasets" name="datasets">
            <RunDatasetsTab />
          </el-tab-pane>

          <el-tab-pane
            v-for="tab in otherSchemaTabs"
            :key="tab.id"
            :label="tab.label"
            :name="tab.id"
          >
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

          <el-tab-pane label="Queue" name="queue">
            <el-card shadow="never" class="queue-card">
              <el-form label-position="top">
                <el-form-item label="GPUs">
                  <el-input-number v-model="numGpus" :min="1" :max="64" />
                </el-form-item>

                <template v-if="showResume">
                  <el-divider content-position="left">Resume</el-divider>
                  <el-form-item>
                    <el-checkbox v-model="fromScratch">
                      Start from scratch (ignore checkpoints)
                    </el-checkbox>
                  </el-form-item>
                  <el-form-item v-if="!fromScratch" label="Resume from checkpoint">
                    <el-select
                      v-model="resumeFrom"
                      placeholder="latest"
                      class="checkpoint-select"
                      :loading="checkpointsLoading"
                    >
                      <el-option
                        v-for="cp in checkpoints"
                        :key="cp.name"
                        :value="cp.name"
                        :label="checkpointLabel(cp)"
                      >
                        <span>{{ checkpointLabel(cp) }}</span>
                        <el-tag v-if="cp.suspect" size="small" type="warning" class="cp-tag">
                          may be corrupt
                        </el-tag>
                      </el-option>
                    </el-select>
                    <p v-if="suspectSelected" class="field-hint field-hint--warn">
                      ⚠ This checkpoint was saved after the last known-good one — it may be truncated
                      (e.g. the disk filled). Prefer the latest good checkpoint if unsure.
                    </p>
                  </el-form-item>
                  <el-empty
                    v-if="!checkpoints.length && !checkpointsLoading"
                    description="No checkpoints found — the run will start from step 0."
                    :image-size="48"
                  />
                </template>

                <el-divider content-position="left">Cache</el-divider>
                <el-form-item>
                  <el-checkbox v-model="cacheOnly">
                    Cache only — build the dataset cache, then exit (no training)
                  </el-checkbox>
                </el-form-item>
                <el-form-item>
                  <el-checkbox v-model="trustCache">
                    Use existing cache (skip the freshness check)
                  </el-checkbox>
                  <el-checkbox v-model="regenerateCache">
                    Regenerate cache (force a full rebuild)
                  </el-checkbox>
                </el-form-item>
              </el-form>
            </el-card>
          </el-tab-pane>

          <el-tab-pane label="TOML" name="toml">
            <div class="toml-pane-toolbar">
              <el-text type="info" size="small">
                Raw TOML. Export bundles the resolved dataset TOMLs (ZIP) for CLI training.
              </el-text>
              <el-button :loading="exporting" @click="exportBundle">Export for CLI (ZIP)</el-button>
            </div>
            <el-input v-model="tomlModel" type="textarea" :rows="24" class="toml-editor" />
          </el-tab-pane>
        </el-tabs>
      </div>
    </ImportTomlOverlay>

    <template #footer>
      <div class="run-form-modal__footer">
        <el-button @click="close">Cancel</el-button>
        <el-button v-if="showSaveForLater" text @click="onSaveForLater">Save for later</el-button>
        <el-button v-if="showSaveChanges" @click="onSaveChanges">Save changes</el-button>
        <el-button type="primary" :loading="submitting" @click="onPrimary">{{ primaryLabel }}</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import { CircleCheck, Upload } from "@element-plus/icons-vue";
import { api } from "../api";
import { downloadBlob } from "../lib/downloadBlob";
import { formatError } from "../lib/formatError";
import ConfigFormSectionCard from "./ConfigFormSectionCard.vue";
import ImportTomlOverlay from "./ImportTomlOverlay.vue";
import RunDatasetsTab from "./RunDatasetsTab.vue";
import {
  buildConfigFormTabs,
  type ConfigSchemaSection,
} from "../lib/configFormSections";
import { sectionHasVisibleFields } from "../lib/configFormSectionLogic";
import { getModelCapability } from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";
import type { CheckpointInfo, JobRecord, TrainingRunRow } from "../types/api";
import type { FormValues, SchemaField } from "../types/forms";

type ModalMode = "create" | "edit" | "continue";

const props = defineProps<{
  modelValue: boolean;
  mode: ModalMode;
  /** Source run for edit/continue modes. */
  job?: TrainingRunRow | null;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "submitted", job: JobRecord): void;
}>();

const vLoading = ElLoadingDirective;
const editor = useConfigEditorStore();
const {
  form,
  schema,
  loading,
  syncing,
  error,
  message,
  validationErrors,
  modelCapabilities,
  continuation,
} = storeToRefs(editor);

const importOverlay = ref<InstanceType<typeof ImportTomlOverlay> | null>(null);
const activeTab = ref("setup");
const submitting = ref(false);
const exporting = ref(false);

// Launch params (kept local; persisted on submit).
const numGpus = ref(1);
const resumeFrom = ref<string>("");
const fromScratch = ref(false);
const cacheOnly = ref(false);
const trustCache = ref(false);
const regenerateCache = ref(false);

const checkpoints = ref<CheckpointInfo[]>([]);
const checkpointsLoading = ref(false);
const editState = ref<string>("");

const tomlModel = computed({
  get: () => editor.content,
  set: (value: string) => editor.setContent(value),
});

const formValues = computed(() => form.value ?? ({} as FormValues));
const selectedCapability = computed(() =>
  getModelCapability(modelCapabilities.value, formValues.value["model.type"])
);
const previewEntryFields = computed(() => {
  const reg = schema.value?.registries as { preview_entry_fields?: SchemaField[] } | undefined;
  return reg?.preview_entry_fields ?? [];
});

const schemaTabs = computed(() => {
  if (!schema.value) return [];
  return buildConfigFormTabs(
    schema.value.sections as ConfigSchemaSection[] | undefined,
    (sec) => sectionHasVisibleFields(sec, formValues.value, modelCapabilities.value)
  );
});

// "Setup" sections, with the dataset field moved to the dedicated Datasets tab.
const setupSections = computed<ConfigSchemaSection[]>(() => {
  const setup = schemaTabs.value.find((t) => t.id === "setup");
  if (!setup) return [];
  return setup.sections.map((sec) =>
    sec.id === "general"
      ? { ...sec, fields: (sec.fields ?? []).filter((f) => f.path !== "dataset") }
      : sec
  );
});
const otherSchemaTabs = computed(() => schemaTabs.value.filter((t) => t.id !== "setup"));

const isCreate = computed(() => props.mode === "create");
const isContinue = computed(() => props.mode === "continue");
const isEdit = computed(() => props.mode === "edit");
const isDraft = computed(() => isEdit.value && editState.value === "new");

const showResume = computed(() => isContinue.value || checkpoints.value.length > 0);
const suspectSelected = computed(
  () => !fromScratch.value && checkpoints.value.find((c) => c.name === resumeFrom.value)?.suspect
);

const showSaveForLater = computed(() => isCreate.value);
const showSaveChanges = computed(() => isDraft.value);
const primaryLabel = computed(() => {
  if (isContinue.value) return "Continue training";
  if (isEdit.value) return isDraft.value ? "Add to queue" : "Save changes";
  return cacheOnly.value ? "Build cache (queue)" : "Add to queue";
});

const title = computed(() => {
  if (isContinue.value) return "Continue training";
  if (isEdit.value) return isDraft.value ? "Edit saved run" : "Edit queued run";
  return "New run";
});

function resetParams(): void {
  numGpus.value = 1;
  resumeFrom.value = "";
  fromScratch.value = false;
  cacheOnly.value = false;
  trustCache.value = false;
  regenerateCache.value = false;
  checkpoints.value = [];
  editState.value = "";
  activeTab.value = "setup";
  submitting.value = false;
}

// trust_cache and regenerate_cache are mutually exclusive.
watch(trustCache, (v) => {
  if (v) regenerateCache.value = false;
});
watch(regenerateCache, (v) => {
  if (v) trustCache.value = false;
});

function checkpointLabel(cp: CheckpointInfo): string {
  const parts = [cp.name, `step ${cp.step}`];
  if (cp.is_latest) parts.push("latest");
  return parts.join(" · ");
}

async function loadCheckpoints(opts: { jobId?: string; runDir?: string }): Promise<void> {
  checkpointsLoading.value = true;
  try {
    const r = opts.jobId
      ? await api.jobCheckpoints(opts.jobId)
      : opts.runDir
        ? await api.runCheckpoints(opts.runDir)
        : null;
    checkpoints.value = r?.checkpoints ?? [];
    const latest = checkpoints.value.find((c) => c.is_latest) ?? checkpoints.value[0];
    if (latest) resumeFrom.value = latest.name;
  } catch {
    checkpoints.value = [];
  } finally {
    checkpointsLoading.value = false;
  }
}

/** Prepare store content + params each time the modal opens. */
async function onOpen(): Promise<void> {
  resetParams();
  await editor.fetchSchema();
  if (isEdit.value && props.job?.job_id) {
    try {
      const j = await api.getJob(String(props.job.job_id));
      editState.value = j.state;
      numGpus.value = j.num_gpus ?? 1;
      cacheOnly.value = !!j.cache_only;
      trustCache.value = !!j.trust_cache;
      regenerateCache.value = !!j.regenerate_cache;
      resumeFrom.value = j.resume_from ?? "";
      await editor.loadContent(j.config_content ?? "");
      if (j.source_run_dir || j.run_dir) {
        await loadCheckpoints({ jobId: String(props.job.job_id) });
        if (j.resume_from) resumeFrom.value = j.resume_from;
      }
    } catch (e) {
      error.value = formatError(e);
    }
  } else if (isContinue.value && props.job?.run_dir) {
    numGpus.value = props.job.num_gpus ?? 1;
    await editor.loadContinuation(props.job.run_dir);
    await loadCheckpoints({ runDir: props.job.run_dir });
  }
  // create mode: the opener pre-loads the store (newConfig or seed content).
}

function onDialogToggle(value: boolean): void {
  emit("update:modelValue", value);
}

function close(): void {
  emit("update:modelValue", false);
}

function onClosed(): void {
  editor.dispose();
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

function onValidate(): void {
  void editor.validateConfig();
}

async function confirmIfInvalid(verb: string): Promise<boolean> {
  try {
    const r = await editor.validateConfig({ quiet: true });
    if (r.ok) return true;
    await ElMessageBox.confirm(
      `${r.errors?.[0] || "Validation failed"}. ${verb} anyway?`,
      "Config not valid",
      { type: "warning", confirmButtonText: `${verb} anyway`, cancelButtonText: "Keep editing" }
    );
    return true;
  } catch (e) {
    if (e === "cancel") return false;
    // a thrown validation error (network) — let the user decide via the alert
    return false;
  }
}

function resumeArg(): string | undefined {
  if (!showResume.value || fromScratch.value) return undefined;
  return resumeFrom.value || undefined;
}

async function doSubmit(action: "queue" | "draft"): Promise<void> {
  if (!(await confirmIfInvalid(action === "draft" ? "Save" : "Queue"))) return;
  submitting.value = true;
  try {
    await editor.flushSync();
    let job: JobRecord;
    if (isContinue.value && continuation.value) {
      job = await api.continueRun({
        run_path: continuation.value.run_dir,
        content: editor.content,
        num_gpus: numGpus.value,
        resume_from: fromScratch.value ? undefined : resumeFrom.value || undefined,
        from_scratch: fromScratch.value,
        enqueue: true,
      });
    } else if (isEdit.value && props.job?.job_id) {
      const id = String(props.job.job_id);
      await api.updateJob(id, {
        content: editor.content,
        num_gpus: numGpus.value,
        resume_from: resumeArg() ?? null,
        cache_only: cacheOnly.value,
        trust_cache: trustCache.value,
        regenerate_cache: regenerateCache.value,
      });
      job = action === "queue" && isDraft.value ? await api.enqueueJob(id) : await api.getJob(id);
    } else {
      job = await api.startJob({
        content: editor.content,
        num_gpus: numGpus.value,
        resume_from: resumeArg(),
        cache_only: cacheOnly.value,
        trust_cache: trustCache.value,
        regenerate_cache: regenerateCache.value,
        enqueue: action === "queue",
        save_for_later: action === "draft",
      });
    }
    ElMessage.success(action === "draft" ? "Saved for later" : "Added to queue");
    emit("submitted", job);
    close();
  } catch (e) {
    error.value = formatError(e);
    ElMessage.error(error.value);
  } finally {
    submitting.value = false;
  }
}

function onPrimary(): void {
  void doSubmit("queue");
}
function onSaveForLater(): void {
  void doSubmit("draft");
}
async function onSaveChanges(): Promise<void> {
  // Edit a draft without enqueuing: persist content + params, keep state "new".
  if (!props.job?.job_id) return;
  submitting.value = true;
  try {
    await editor.flushSync();
    const id = String(props.job.job_id);
    await api.updateJob(id, {
      content: editor.content,
      num_gpus: numGpus.value,
      resume_from: resumeArg() ?? null,
      cache_only: cacheOnly.value,
      trust_cache: trustCache.value,
      regenerate_cache: regenerateCache.value,
    });
    const job = await api.getJob(id);
    ElMessage.success("Saved");
    emit("submitted", job);
    close();
  } catch (e) {
    error.value = formatError(e);
    ElMessage.error(error.value);
  } finally {
    submitting.value = false;
  }
}

async function exportBundle(): Promise<void> {
  const text = (editor.content || "").trim();
  if (!text) {
    ElMessage.warning("Nothing to export — add config TOML first.");
    return;
  }
  exporting.value = true;
  try {
    await editor.flushSync();
    const name = editor.editorRunName || "training_export";
    const { blob, filename } = await api.exportConfigBundle(name, editor.content);
    downloadBlob(blob, filename);
    ElMessage.success("Exported ZIP for CLI");
  } catch (e) {
    error.value = formatError(e);
    ElMessage.error(error.value);
  } finally {
    exporting.value = false;
  }
}

// React to open transitions (the opener flips modelValue and sets mode/job + store content).
watch(
  () => props.modelValue,
  (open) => {
    if (open) void onOpen();
  }
);
</script>

<style scoped>
.run-form-modal :deep(.el-dialog__body) {
  padding-top: 8px;
}
.run-form-modal__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.run-form-modal__title {
  font-size: 18px;
  font-weight: 600;
}
.run-form-modal__header-actions {
  display: flex;
  gap: 8px;
}
.run-form-modal__body {
  min-height: 200px;
}
.run-form-modal__footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.mb-12 {
  margin-bottom: 12px;
}
.tab-sections {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.tab-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.queue-card {
  max-width: 640px;
  border: 1px solid var(--el-border-color-lighter);
}
.checkpoint-select {
  width: 100%;
  max-width: 420px;
}
.cp-tag {
  margin-left: 8px;
}
.field-hint {
  margin: 6px 0 0;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.4;
}
.field-hint--warn {
  color: var(--el-color-warning);
}
.continue-text {
  font-size: 13px;
  line-height: 1.5;
}
.validation-alert :deep(.el-alert__content) {
  width: 100%;
}
.validation-errors {
  margin: 8px 0 0;
  padding-left: 1.25rem;
  line-height: 1.5;
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

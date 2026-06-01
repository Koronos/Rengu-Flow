<template>
  <div class="page-shell run-form-view">
    <div class="page-head run-form-view__head">
      <el-button :icon="ArrowLeft" @click="goBack">Runs</el-button>
      <span class="run-form-view__title">{{ title }}</span>
      <div class="run-form-view__head-actions">
        <el-button :icon="CircleCheck" @click="onValidate">Validate</el-button>
        <el-button :icon="Upload" @click="triggerImport">Import TOML…</el-button>
        <el-button v-if="showSaveForLater" @click="onSaveForLater">Save for later</el-button>
        <el-button v-if="showSaveChanges" @click="onSaveChanges">Save changes</el-button>
        <el-button type="primary" :loading="submitting" @click="onPrimary">{{ primaryLabel }}</el-button>
      </div>
    </div>

    <ImportTomlOverlay ref="importOverlay" @import="handleImport">
      <div v-loading="loading || (syncing && !form)" class="run-form-view__body">
        <div class="run-form-view__name-row">
          <el-input
            v-model="runNameModel"
            class="run-name-input"
            placeholder="Run name — optional; a timestamp folder is used if empty"
            maxlength="120"
            show-word-limit
          />
        </div>

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
            Run tab (or start from scratch). Raise <code>epochs</code>/<code>max_steps</code> to train further.
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

          <el-tab-pane label="Run" name="queue">
            <div class="tab-sections">
              <el-card v-if="runConfigFields.length" shadow="never" class="run-section-card">
                <template #header><span class="run-section-title">Run configuration</span></template>
                <el-form label-position="top">
                  <ConfigFormField
                    v-for="f in runConfigFields"
                    :key="f.path"
                    :field="f"
                    :form="formValues"
                    :capabilities="modelCapabilities"
                    @update:path="onFieldUpdate"
                  />
                </el-form>
              </el-card>

              <el-card shadow="never" class="run-section-card">
                <template #header><span class="run-section-title">Compute</span></template>
                <el-form label-position="top">
                  <el-form-item>
                    <template #label>
                      <span class="launch-label">
                        <span>GPUs</span>
                        <FieldHelpIcon :field="(HELP.numGpus as unknown as SchemaField)" />
                        <code class="cli-flag">deepspeed --num_gpus</code>
                      </span>
                    </template>
                    <el-input-number v-model="numGpus" :min="1" :max="64" />
                  </el-form-item>
                </el-form>
              </el-card>

              <el-card v-if="showResume" shadow="never" class="run-section-card">
                <template #header><span class="run-section-title">Resume</span></template>
                <el-form label-position="top">
                  <el-text type="info" size="small" class="resume-hint">
                    Picks the checkpoint this launch resumes from
                    (<code>--resume_from_checkpoint</code>).
                  </el-text>
                  <el-form-item>
                    <el-checkbox v-model="fromScratch">
                      Start from scratch (ignore checkpoints)
                    </el-checkbox>
                  </el-form-item>
                  <el-form-item v-if="!fromScratch">
                    <template #label>
                      <span class="launch-label">
                        <span>Resume from checkpoint</span>
                        <FieldHelpIcon :field="(HELP.resumeFrom as unknown as SchemaField)" />
                        <code class="cli-flag">--resume_from_checkpoint</code>
                      </span>
                    </template>
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
                </el-form>
              </el-card>

              <el-card shadow="never" class="run-section-card">
                <template #header><span class="run-section-title">Cache</span></template>
                <el-form label-position="top">
                  <el-form-item>
                    <span class="launch-label">
                      <el-checkbox v-model="cacheOnly">
                        Cache only — build the dataset cache, then exit (no training)
                      </el-checkbox>
                      <FieldHelpIcon :field="(HELP.cacheOnly as unknown as SchemaField)" />
                      <code class="cli-flag">--cache_only</code>
                    </span>
                  </el-form-item>
                  <el-form-item class="cache-toggles">
                    <span class="launch-label">
                      <el-checkbox v-model="trustCache">
                        Use existing cache (skip the freshness check)
                      </el-checkbox>
                      <FieldHelpIcon :field="(HELP.trustCache as unknown as SchemaField)" />
                      <code class="cli-flag">--trust_cache</code>
                    </span>
                    <span class="launch-label">
                      <el-checkbox v-model="regenerateCache">
                        Regenerate cache (force a full rebuild)
                      </el-checkbox>
                      <FieldHelpIcon :field="(HELP.regenerateCache as unknown as SchemaField)" />
                      <code class="cli-flag">--regenerate_cache</code>
                    </span>
                  </el-form-item>
                </el-form>
              </el-card>
            </div>
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
  </div>
</template>

<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { storeToRefs } from "pinia";
import { ElLoadingDirective, ElMessage, ElMessageBox } from "element-plus";
import { ArrowLeft, CircleCheck, Upload } from "@element-plus/icons-vue";
import { api } from "../api";
import { downloadBlob } from "../lib/downloadBlob";
import { formatError } from "../lib/formatError";
import ConfigFormField from "../components/ConfigFormField.vue";
import ConfigFormSectionCard from "../components/ConfigFormSectionCard.vue";
import FieldHelpIcon from "../components/FieldHelpIcon.vue";
import ImportTomlOverlay from "../components/ImportTomlOverlay.vue";
import RunDatasetsTab from "../components/RunDatasetsTab.vue";
import {
  buildConfigFormTabs,
  type ConfigSchemaSection,
} from "../lib/configFormSections";
import { sectionHasVisibleFields } from "../lib/configFormSectionLogic";
import { getModelCapability } from "../lib/formUtils";
import { useConfigEditorStore } from "../stores/configEditor";
import type { CheckpointInfo } from "../types/api";
import type { FormValues, SchemaField } from "../types/forms";

const route = useRoute();
const router = useRouter();
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

// Mode is derived from the route: edit has a job id; continue carries ?continue_run=<dir>.
const mode = computed<"create" | "edit" | "continue">(() => {
  if (route.name === "run-edit") return "edit";
  if (typeof route.query.continue_run === "string" && route.query.continue_run) return "continue";
  return "create";
});
const routeJobId = computed(() => String(route.params.id || ""));
const continueDir = computed(() =>
  typeof route.query.continue_run === "string" ? route.query.continue_run : ""
);

const importOverlay = ref<InstanceType<typeof ImportTomlOverlay> | null>(null);
const activeTab = ref("setup");
const submitting = ref(false);
const exporting = ref(false);

// Help/CLI annotations for the hand-built launch controls (mirrors how
// schema-driven config fields show a help icon + a monospace flag/path hint).
const HELP = {
  numGpus: {
    path: "num_gpus",
    help: "Number of GPUs for the DeepSpeed launcher for this run.",
    doc_path: "docs/user/cli.md",
  },
  cacheOnly: {
    help: "Build the dataset cache (latents + text embeddings) and exit without training.",
    doc_path: "docs/developer/dataset-and-cache.md",
  },
  trustCache: {
    help: "Skip the cache freshness check and reuse the existing cache as-is.",
    doc_path: "docs/developer/dataset-and-cache.md",
  },
  regenerateCache: {
    help: "Force a full rebuild of the dataset cache.",
    doc_path: "docs/developer/dataset-and-cache.md",
  },
  resumeFrom: {
    help: "Checkpoint folder this launch resumes from (overrides the resume_from_checkpoint config value).",
    doc_path: "docs/user/cli.md",
  },
} as const;

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
const runNameModel = computed({
  get: () => (typeof formValues.value.run_name === "string" ? (formValues.value.run_name as string) : ""),
  set: (v: string) => editor.patchFormField("run_name", v),
});
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

// Config fields moved out of Setup into the Run tab (launch/run-execution settings).
const RUN_CONFIG_PATHS = ["output_dir", "resume_from_checkpoint"];

// "Setup" sections, with the dataset field moved to the Datasets tab and the
// run-execution fields (output_dir, resume_from_checkpoint) moved to the Run tab.
const setupSections = computed<ConfigSchemaSection[]>(() => {
  const setup = schemaTabs.value.find((t) => t.id === "setup");
  if (!setup) return [];
  return setup.sections
    .map((sec) =>
      sec.id === "general"
        ? {
            ...sec,
            // dataset -> Datasets tab; output_dir/resume_from_checkpoint -> Run tab;
            // run_name -> the prominent name input above the tabs.
            fields: (sec.fields ?? []).filter(
              (f) =>
                f.path !== "dataset" &&
                f.path !== "run_name" &&
                !RUN_CONFIG_PATHS.includes(f.path)
            ),
          }
        : sec
    )
    .filter((sec) => (sec.fields ?? []).length > 0);
});
const otherSchemaTabs = computed(() => schemaTabs.value.filter((t) => t.id !== "setup"));

// Run-execution config fields shown in the Run tab. `resume_from_checkpoint` is intentionally
// NOT shown here — resuming is controlled by the checkpoint selector below (it sets the
// `--resume_from_checkpoint` launch flag); the boolean config value stays editable in the TOML tab.
const runConfigFields = computed<SchemaField[]>(() => {
  const setup = schemaTabs.value.find((t) => t.id === "setup");
  const general = setup?.sections.find((sec) => sec.id === "general");
  return (general?.fields ?? []).filter((f) => f.path === "output_dir");
});

const isCreate = computed(() => mode.value === "create");
const isContinue = computed(() => mode.value === "continue");
const isEdit = computed(() => mode.value === "edit");
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

function onFieldUpdate({ path, value }: { path: string; value: unknown }): void {
  editor.patchFormField(path, value);
}

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

/**
 * Load the run being edited/continued. For "create", the opener (Runs list) prepares the store
 * (blank via newConfig, or seeded via loadContent) before navigating here, so we leave it alone.
 */
async function init(): Promise<void> {
  resetParams();
  await editor.fetchSchema();
  if (isEdit.value && routeJobId.value) {
    try {
      const j = await api.getJob(routeJobId.value);
      editState.value = j.state;
      numGpus.value = j.num_gpus ?? 1;
      cacheOnly.value = !!j.cache_only;
      trustCache.value = !!j.trust_cache;
      regenerateCache.value = !!j.regenerate_cache;
      resumeFrom.value = j.resume_from ?? "";
      await editor.loadContent(j.config_content ?? "");
      if (j.source_run_dir || j.run_dir) {
        await loadCheckpoints({ jobId: routeJobId.value });
        if (j.resume_from) resumeFrom.value = j.resume_from;
      }
    } catch (e) {
      error.value = formatError(e);
    }
  } else if (isContinue.value && continueDir.value) {
    await editor.loadContinuation(continueDir.value);
    await loadCheckpoints({ runDir: continueDir.value });
  }
}

watch(() => route.fullPath, init, { immediate: true });

onUnmounted(() => {
  editor.dispose();
});

function goBack(): void {
  router.push({ name: "jobs" });
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
    if (isContinue.value && continuation.value) {
      await api.continueRun({
        run_path: continuation.value.run_dir,
        content: editor.content,
        num_gpus: numGpus.value,
        resume_from: fromScratch.value ? undefined : resumeFrom.value || undefined,
        from_scratch: fromScratch.value,
        enqueue: true,
      });
    } else if (isEdit.value && routeJobId.value) {
      const id = routeJobId.value;
      await api.updateJob(id, {
        content: editor.content,
        num_gpus: numGpus.value,
        resume_from: resumeArg() ?? null,
        cache_only: cacheOnly.value,
        trust_cache: trustCache.value,
        regenerate_cache: regenerateCache.value,
      });
      if (action === "queue" && isDraft.value) await api.enqueueJob(id);
    } else {
      await api.startJob({
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
    goBack();
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
  if (!routeJobId.value) return;
  submitting.value = true;
  try {
    await editor.flushSync();
    await api.updateJob(routeJobId.value, {
      content: editor.content,
      num_gpus: numGpus.value,
      resume_from: resumeArg() ?? null,
      cache_only: cacheOnly.value,
      trust_cache: trustCache.value,
      regenerate_cache: regenerateCache.value,
    });
    ElMessage.success("Saved");
    goBack();
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
</script>

<style scoped>
.run-form-view__head {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  padding: var(--rf-space-sm) 0;
  background: var(--el-bg-color);
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.run-form-view__title {
  font-size: 18px;
  font-weight: 600;
  flex: 1;
  min-width: 120px;
}
.run-form-view__head-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.run-form-view__body {
  min-height: 200px;
  margin-top: var(--rf-space-sm);
}
.run-form-view__name-row {
  margin-bottom: var(--rf-space-sm);
}
.run-name-input {
  max-width: 520px;
}
.run-name-input :deep(.el-input__inner) {
  font-size: 18px;
  font-weight: 600;
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
.run-section-card {
  border: 1px solid var(--el-border-color-lighter);
}
.run-section-card :deep(.el-card__header) {
  padding: 12px 16px;
}
.run-section-card :deep(.el-card__body) {
  padding: 12px 16px 16px;
}
.run-section-title {
  font-weight: 600;
}
.checkpoint-select {
  width: 100%;
  max-width: 420px;
}
.launch-label {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
.cli-flag {
  font-family: ui-monospace, monospace;
  font-size: 11px;
  color: var(--el-text-color-secondary);
}
.cache-toggles :deep(.el-form-item__content) {
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}
.resume-hint {
  display: block;
  margin: -4px 0 10px;
  line-height: 1.4;
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

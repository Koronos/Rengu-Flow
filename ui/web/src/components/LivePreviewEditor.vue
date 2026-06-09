<template>
  <div class="live-preview-editor">
    <div class="lpe-head">
      <el-text type="info" size="small">
        Edit previews of the running run and apply without restarting. Persists to the
        run's config (survives resume). Only the [preview] section is hot-reloaded.
      </el-text>
      <el-button size="small" link :icon="Refresh" :loading="loading" @click="load">
        Reload
      </el-button>
    </div>

    <el-form label-position="top" class="lpe-form">
      <el-form-item>
        <template #label>
          <span class="lpe-switch-label">
            <el-switch :model-value="enabled" @update:model-value="(v) => (enabled = Boolean(v))" />
            <span>{{ enabled ? "Previews enabled" : "Previews disabled" }}</span>
          </span>
        </template>
      </el-form-item>

      <el-form-item label="Sampling prompts">
        <PreviewEntriesField
          :model-value="prompts"
          :entry-fields="previewEntryFields"
          :parent-form="parentForm"
          :capabilities="capabilities"
          @update:model-value="(v) => (prompts = v)"
        />
      </el-form-item>

      <div class="lpe-grid">
        <el-form-item label="Every N steps">
          <el-input-number v-model="everyNSteps" :min="1" :step="1" controls-position="right" />
        </el-form-item>
        <el-form-item label="Every N epochs">
          <el-input-number v-model="everyNEpochs" :min="1" :step="1" controls-position="right" />
        </el-form-item>
        <el-form-item label="Width">
          <el-input-number v-model="width" :min="64" :step="64" controls-position="right" />
        </el-form-item>
        <el-form-item label="Height">
          <el-input-number v-model="height" :min="64" :step="64" controls-position="right" />
        </el-form-item>
        <el-form-item label="Inference steps">
          <el-input-number v-model="steps" :min="1" :step="1" controls-position="right" />
        </el-form-item>
        <el-form-item label="Guidance scale">
          <el-input-number v-model="guidance" :min="0" :step="0.5" controls-position="right" />
        </el-form-item>
        <el-form-item label="Seed">
          <el-input-number v-model="seed" :step="1" controls-position="right" />
        </el-form-item>
      </div>

      <el-form-item label="Negative prompt">
        <el-input v-model="negativePrompt" placeholder="(optional)" />
      </el-form-item>

      <div class="lpe-actions">
        <el-button :loading="previewing" @click="previewNow">Preview now</el-button>
        <el-button :loading="saving" @click="apply(false)">Apply</el-button>
        <el-button type="primary" :loading="saving" @click="apply(true)">
          Apply &amp; preview now
        </el-button>
      </div>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import { Refresh } from "@element-plus/icons-vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import PreviewEntriesField from "./PreviewEntriesField.vue";
import { normalizePreviewEntries, type PreviewEntry } from "../lib/previewEntries";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

const props = defineProps<{ jobId: string | number }>();

const loading = ref(false);
const saving = ref(false);
const previewing = ref(false);
/** The full [preview] table as last loaded — unknown keys are preserved on save. */
const original = ref<Record<string, unknown>>({});

// The schema feeds the shared PreviewEntriesField the same per-prompt fields the config form
// uses, so live edits keep prompt names/titles and per-prompt overrides (no more "prompt 0/1").
const schema = ref<Record<string, unknown> | null>(null);
const modelType = ref("");
const previewEntryFields = computed(
  () => (schema.value?.registries as { preview_entry_fields?: SchemaField[] } | undefined)
    ?.preview_entry_fields ?? []
);
const capabilities = computed<ModelCapabilities>(
  () => (schema.value?.registries as { model_capabilities?: ModelCapabilities } | undefined)
    ?.model_capabilities ?? {}
);
const parentForm = computed<FormValues>(() => ({ "model.type": modelType.value }));

const enabled = ref(true);
const prompts = ref<PreviewEntry[]>([]);
const everyNSteps = ref<number | undefined>(undefined);
const everyNEpochs = ref<number | undefined>(undefined);
const width = ref<number | undefined>(undefined);
const height = ref<number | undefined>(undefined);
const steps = ref<number | undefined>(undefined);
const guidance = ref<number | undefined>(undefined);
const seed = ref<number | undefined>(undefined);
const negativePrompt = ref("");

function numOrUndef(v: unknown): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function populate(preview: Record<string, unknown>): void {
  original.value = preview || {};
  enabled.value = preview.enabled !== false;
  prompts.value = normalizePreviewEntries(preview.prompts);
  everyNSteps.value = numOrUndef(preview.preview_every_n_steps);
  everyNEpochs.value = numOrUndef(preview.preview_every_n_epochs);
  width.value = numOrUndef(preview.width);
  height.value = numOrUndef(preview.height);
  steps.value = numOrUndef(preview.num_inference_steps);
  guidance.value = numOrUndef(preview.guidance_scale);
  seed.value = numOrUndef(preview.seed);
  negativePrompt.value = typeof preview.negative_prompt === "string" ? preview.negative_prompt : "";
}

async function ensureSchema(): Promise<void> {
  if (schema.value) return;
  try {
    schema.value = (await api.getSchema()) as Record<string, unknown>;
  } catch {
    // Without the schema the per-prompt fields degrade gracefully; prompt names still render.
  }
}

async function load(): Promise<void> {
  if (!props.jobId) return;
  loading.value = true;
  try {
    void ensureSchema();
    const r = await api.getJobPreviewConfig(String(props.jobId));
    modelType.value = r.model_type || "";
    populate(r.preview || {});
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    loading.value = false;
  }
}

function buildPreview(): Record<string, unknown> {
  // Merge over the loaded table so keys we don't expose (e.g. seed_stride, save_png,
  // model-specific options) are preserved.
  const next: Record<string, unknown> = { ...original.value };
  next.enabled = enabled.value;
  next.prompts = prompts.value;
  const setOrDelete = (key: string, val: number | undefined) => {
    if (val === undefined || val === null || Number.isNaN(val)) delete next[key];
    else next[key] = val;
  };
  setOrDelete("preview_every_n_steps", everyNSteps.value);
  setOrDelete("preview_every_n_epochs", everyNEpochs.value);
  setOrDelete("width", width.value);
  setOrDelete("height", height.value);
  setOrDelete("num_inference_steps", steps.value);
  setOrDelete("guidance_scale", guidance.value);
  setOrDelete("seed", seed.value);
  const neg = negativePrompt.value.trim();
  if (neg) next.negative_prompt = neg;
  else delete next.negative_prompt;
  return next;
}

async function previewNow(): Promise<void> {
  if (!props.jobId) return;
  previewing.value = true;
  try {
    // Just trigger a preview with the current (last-applied) settings — runs even if
    // previews are disabled, as long as there are prompts.
    await api.sendJobSignal(String(props.jobId), "preview");
    ElMessage.success("Rendering a preview…");
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    previewing.value = false;
  }
}

async function apply(withPreview: boolean): Promise<void> {
  if (!props.jobId) return;
  saving.value = true;
  try {
    await api.updateJobPreviewConfig(String(props.jobId), buildPreview(), withPreview);
    ElMessage.success(
      withPreview ? "Preview settings applied; rendering a preview…" : "Preview settings applied"
    );
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    saving.value = false;
  }
}

watch(() => props.jobId, load, { immediate: true });
</script>

<style scoped>
.live-preview-editor {
  width: 100%;
}
.lpe-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.lpe-switch-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.lpe-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0 16px;
}
.lpe-actions {
  display: flex;
  gap: 8px;
  margin-top: 4px;
}
</style>

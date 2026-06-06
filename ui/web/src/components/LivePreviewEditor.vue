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

      <el-form-item label="Prompts (one per line)">
        <el-input
          v-model="promptsText"
          type="textarea"
          :rows="4"
          placeholder="a cat sitting on a sofa&#10;a watercolor landscape"
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

const props = defineProps<{ jobId: string | number }>();

const loading = ref(false);
const saving = ref(false);
/** The full [preview] table as last loaded — unknown keys are preserved on save. */
const original = ref<Record<string, unknown>>({});

const enabled = ref(true);
const promptsText = ref("");
const everyNSteps = ref<number | undefined>(undefined);
const everyNEpochs = ref<number | undefined>(undefined);
const width = ref<number | undefined>(undefined);
const height = ref<number | undefined>(undefined);
const steps = ref<number | undefined>(undefined);
const guidance = ref<number | undefined>(undefined);
const seed = ref<number | undefined>(undefined);
const negativePrompt = ref("");

function promptsToText(prompts: unknown): string {
  if (!Array.isArray(prompts)) return "";
  return prompts
    .map((p) => {
      if (typeof p === "string") return p;
      if (p && typeof p === "object") {
        const o = p as Record<string, unknown>;
        return String(o.prompt ?? o.text ?? "");
      }
      return "";
    })
    .filter((s) => s.trim() !== "")
    .join("\n");
}

function numOrUndef(v: unknown): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function populate(preview: Record<string, unknown>): void {
  original.value = preview || {};
  enabled.value = preview.enabled !== false;
  promptsText.value = promptsToText(preview.prompts);
  everyNSteps.value = numOrUndef(preview.preview_every_n_steps);
  everyNEpochs.value = numOrUndef(preview.preview_every_n_epochs);
  width.value = numOrUndef(preview.width);
  height.value = numOrUndef(preview.height);
  steps.value = numOrUndef(preview.num_inference_steps);
  guidance.value = numOrUndef(preview.guidance_scale);
  seed.value = numOrUndef(preview.seed);
  negativePrompt.value = typeof preview.negative_prompt === "string" ? preview.negative_prompt : "";
}

async function load(): Promise<void> {
  if (!props.jobId) return;
  loading.value = true;
  try {
    const r = await api.getJobPreviewConfig(String(props.jobId));
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
  const prompts = promptsText.value
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s !== "");
  next.prompts = prompts;
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

async function apply(previewNow: boolean): Promise<void> {
  if (!props.jobId) return;
  saving.value = true;
  try {
    await api.updateJobPreviewConfig(String(props.jobId), buildPreview(), previewNow);
    ElMessage.success(
      previewNow ? "Preview settings applied; rendering a preview…" : "Preview settings applied"
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

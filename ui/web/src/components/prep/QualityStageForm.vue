<!--
  Quality filter options, extracted verbatim from PrepJobFormView.vue.

  Owns the inline report-only preview run: it queues a real `quality` job with
  `action: "report"`, follows it over the live channel and renders the report.
  The button that triggers it lives in the view's action row, so `runPreview` is
  exposed and `previewRunning` is surfaced as a v-model.
-->
<template>
  <h3 class="section-title">Quality filter options</h3>
  <el-alert
    type="info"
    :closable="false"
    show-icon
    class="mt-8 mb-12"
    title="Scans images for blur or low aesthetic quality and flags (or moves) them. Non-destructive by default — report mode only lists flagged images without moving them."
  />
  <el-form label-position="top" :disabled="disabled">
    <el-form-item>
      <template #label>
        Metric <FieldHelpIcon :field="help('Blur: fast Laplacian + resolution heuristic, no download. Aesthetic: anime booru appeal model (worst→masterpiece), downloads on first use. Technical IQA: learned No-Reference model scoring perceived technical quality (blur, noise, compression) — works on anime and natural photos, downloads a model on first use.')" />
        <FieldPathTag path="quality.metric" />
      </template>
      <el-select v-model="model.metric" class="w-full">
        <el-option label="Blur / resolution (fast, no download)" value="blur" />
        <el-option label="Aesthetic — booru quality (deepghs, downloads a model)" value="aesthetic" />
        <el-option label="Technical IQA — image quality (pyiqa, anime + photo, downloads a model)" value="iqa" />
      </el-select>
    </el-form-item>

    <template v-if="model.metric === 'blur'">
      <el-form-item>
        <template #label>
          Blur threshold <FieldHelpIcon :field="help('Laplacian-variance floor measured on a long-side-512 copy; images below it are flagged blurry (default 80). Run in report mode first, then set the threshold between your good and bad samples.')" />
          <FieldPathTag path="quality.blur_threshold" />
        </template>
        <el-input-number
          v-model="model.blur_threshold"
          :min="0"
          placeholder="80"
          controls-position="right"
        />
      </el-form-item>

      <el-form-item>
        <template #label>
          Min side (px) <FieldHelpIcon :field="help('Flag images whose shorter side (pixels) is below this value (default 0 = off). Use it to catch undersized images that would degrade training resolution buckets.')" />
          <FieldPathTag path="quality.min_side" />
        </template>
        <el-input-number
          v-model="model.min_side"
          :min="0"
          placeholder="0"
          controls-position="right"
        />
      </el-form-item>

      <el-form-item>
        <template #label>
          Min detail <FieldHelpIcon :field="help('Effective-resolution floor: flags pixelated or upscaled images (a 64px picture blown up to 1024 still reads as sharp to the blur check, but has little real detail). 0 disables it. Typical starting point ~12–15; calibrate in report mode.')" />
          <FieldPathTag path="quality.min_detail" />
        </template>
        <el-input-number
          v-model="model.min_detail"
          :min="0"
          :step="0.5"
          placeholder="0"
          controls-position="right"
        />
      </el-form-item>
    </template>

    <el-form-item v-if="model.metric === 'aesthetic'">
      <template #label>
        Minimum label <FieldHelpIcon :field="help('Flags any image the booru-quality model ranks below this tier (default normal). Higher = stricter; moving up to good will flag the worst, low, and normal tiers.')" />
        <FieldPathTag path="quality.aesthetic_min_label" />
      </template>
      <el-select v-model="model.aesthetic_min_label" class="w-full">
        <el-option label="worst — flag nothing" value="worst" />
        <el-option label="low — flag worst" value="low" />
        <el-option label="normal — flag worst &amp; low" value="normal" />
        <el-option label="good — flag worst, low &amp; normal" value="good" />
        <el-option label="great — flag everything below great" value="great" />
        <el-option label="best — flag everything below best" value="best" />
        <el-option label="masterpiece — flag everything below masterpiece" value="masterpiece" />
      </el-select>
    </el-form-item>

    <template v-if="model.metric === 'iqa'">
      <el-form-item>
        <template #label>
          Model <FieldHelpIcon :field="help('Which NR-IQA model scores quality. clipiqa/arniqa generalize to illustration; musiq/maniqa are tuned on natural photos; brisque/niqe are classic baselines.')" />
          <FieldPathTag path="quality.iqa_model" />
        </template>
        <el-select v-model="model.iqa_model" class="w-full">
          <el-option label="CLIP-IQA — any domain (anime + photo)" value="clipiqa" />
          <el-option label="ARNIQA — any domain, robust" value="arniqa" />
          <el-option label="MUSIQ — natural photos" value="musiq" />
          <el-option label="MANIQA — natural photos" value="maniqa" />
          <el-option label="BRISQUE — classic, photos" value="brisque" />
          <el-option label="NIQE — classic, opinion-free" value="niqe" />
        </el-select>
      </el-form-item>

      <el-form-item>
        <template #label>
          Discard lowest % <FieldHelpIcon :field="help('Flags the lowest-quality N% of the dataset, ranked by the selected model (same behavior for every model). 10 = drop the worst 10%; raise it to cull more. Run report mode first to see how many fall at each level.')" />
          <FieldPathTag path="quality.iqa_threshold" />
        </template>
        <el-slider
          v-model="model.iqa_threshold"
          :min="0"
          :max="100"
          :step="1"
          show-input
          class="w-full"
        />
      </el-form-item>
    </template>

    <el-form-item>
      <template #label>
        Move flagged <FieldHelpIcon :field="help('Moves flagged images into &lt;path&gt;/low_quality/ (off = report only, non-destructive). Use report mode first to review what would be flagged before enabling move.')" />
        <FieldPathTag path="quality.action" />
      </template>
      <el-switch v-model="model.move" />
      <el-text class="ml-8" size="small">Move flagged images into &lt;path&gt;/low_quality (off = report only)</el-text>
    </el-form-item>

    <el-form-item v-if="model.move">
      <template #label>
        Output directory <FieldHelpIcon :field="help('Where to move flagged images. Defaults to &lt;path&gt;/low_quality.')" />
        <FieldPathTag path="quality.output_dir" />
      </template>
      <PathFieldControl
        v-model="model.output_dir"
        expect="dir"
        placeholder="e.g. /data/rejects (default: <path>/low_quality)"
        input-class="w-full"
      />
    </el-form-item>
  </el-form>

  <!-- Inline quality report preview -->
  <div v-if="previewJobId" class="preview-report-panel mt-8">
    <div v-if="previewRunning" class="preview-progress">
      <el-text size="small" class="hint-text">{{ previewProgress?.msg || 'Running…' }}</el-text>
      <el-progress :percentage="Math.min(100, Math.round(previewProgress?.percent ?? 0))" :show-text="false" class="mt-8" />
    </div>
    <el-alert v-if="previewError" type="error" :title="previewError" show-icon :closable="false" class="mt-8" />
    <template v-if="previewReport">
      <h3 class="section-title mt-8">Report (nothing was changed — report only)</h3>
      <el-descriptions :column="3" border size="small">
        <el-descriptions-item
          v-for="(val, key) in previewReportFields"
          :key="key"
          :label="String(key)"
        >{{ val }}</el-descriptions-item>
      </el-descriptions>
      <el-text
        v-if="Number(previewReport.flagged) > 0"
        size="small"
        class="hint-text preview-flagged mt-8"
      >
        {{ previewReport.flagged }} of {{ previewReport.scored }} images would be flagged
      </el-text>
      <el-text
        v-if="previewReasonsSummary"
        size="small"
        class="hint-text preview-reasons mt-4"
      >
        {{ previewReasonsSummary }}
      </el-text>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import type { PropType } from "vue";
import { api } from "../../api";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import PathFieldControl from "../PathFieldControl.vue";
import { copyKnown, help } from "./formHelpers";
import { formatError } from "../../lib/formatError";
import { usePrepJobLive } from "../../composables/usePrepJobLive";
import { buildStageConfig } from "../../lib/prepStageConfig";
import type { PrepCommonForm, PrepQualityForm } from "../../lib/prepStageConfig";
import type { PrepQualityConfig } from "../../types/api";

const model = defineModel<PrepQualityForm>({ required: true });
/** Surfaced upward: the view's "Preview report" button shows this as its loading state. */
const previewRunning = defineModel<boolean>("previewRunning", { default: false });

const props = defineProps({
  /** Needed to queue the preview job (path + caption layout). */
  commonForm: { type: Object as PropType<PrepCommonForm>, required: true },
  /** `quality` section of a cloned job config. */
  seed: { type: Object as PropType<PrepQualityConfig | null>, default: null },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the whole stage form.
   */
  disabled: { type: Boolean, default: false },
});

// --- quality inline preview (report-only) ---
const previewJobId = ref<string | null>(null);
const previewReport = ref<Record<string, unknown> | null>(null);
const previewError = ref("");

const { progress: previewProgress } = usePrepJobLive(
  () => previewJobId.value ?? undefined,
  { onRunFinished: onPreviewFinished }
);

const PREVIEW_EXCLUDED = new Set(["failed", "errors", "low_quality"]);
const previewReportFields = computed(() => {
  if (!previewReport.value) return {} as Record<string, unknown>;
  return Object.fromEntries(
    Object.entries(previewReport.value).filter(([k]) => !PREVIEW_EXCLUDED.has(k))
  ) as Record<string, unknown>;
});

const previewReasonsSummary = computed(() => {
  const lq = previewReport.value?.low_quality;
  if (!Array.isArray(lq) || lq.length === 0) return "";
  const counts: Record<string, number> = {};
  for (const item of lq) {
    const reasons = (item as Record<string, unknown>).reasons;
    if (Array.isArray(reasons)) {
      for (const r of reasons) {
        const s = String(r);
        counts[s] = (counts[s] ?? 0) + 1;
      }
    }
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([r, n]) => `${r}: ${n}`)
    .join(" · ");
});

async function onPreviewFinished(): Promise<void> {
  const id = previewJobId.value;
  if (!id) return;
  try {
    const res = await api.prepJobReport(id);
    previewReport.value = res.report;
  } catch {
    // report.json may lag the finish signal
    await new Promise<void>((r) => setTimeout(r, 600));
    try {
      const res = await api.prepJobReport(id);
      previewReport.value = res.report;
    } catch (e) {
      previewError.value = formatError(e);
    }
  }
  previewRunning.value = false;
}

async function runPreview(): Promise<void> {
  previewError.value = "";
  previewReport.value = null;
  if (!props.commonForm.path.trim()) {
    previewError.value = "Dataset folder is required.";
    return;
  }
  previewRunning.value = true;
  try {
    const job = await api.createPrepJob({
      stage: "quality",
      config: buildStageConfig("quality", { form: props.commonForm, qualityForm: model.value }),
      start_now: true,
    });
    previewJobId.value = String(job.id);
  } catch (e) {
    previewError.value = formatError(e);
    previewRunning.value = false;
  }
}

let seedApplied = false;

function applySeed(): void {
  const seed = props.seed;
  if (!seed || seedApplied) return;
  seedApplied = true;
  copyKnown(model.value as unknown as Record<string, unknown>, seed);
  // action -> move conversion (copyKnown skips "action" since the form field is "move")
  if (seed.action !== undefined) model.value.move = seed.action === "move";
}

watch(() => props.seed, applySeed);
onMounted(applySeed);

defineExpose({ runPreview });
</script>

<style scoped>
.section-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.hint-text {
  display: block;
  margin-top: 4px;
  margin-bottom: 8px;
}
.ml-8 {
  margin-left: 8px;
}
.mt-8 {
  margin-top: 8px;
}
.mt-4 {
  margin-top: 4px;
}
.preview-report-panel {
  border-top: 1px solid var(--el-border-color-lighter);
  padding-top: 16px;
}
.preview-progress {
  padding: 4px 0;
}
.preview-flagged,
.preview-reasons {
  display: block;
}
</style>

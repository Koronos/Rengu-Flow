<template>
  <div class="qi-view">
    <div class="page-head qi-view__head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Dataset Studio</el-button>
      <span class="qi-view__title">Quality index</span>
    </div>

    <el-card shadow="never" class="qi-view__setup">
      <template #header>Dataset and models</template>
      <p class="page-hint">
        Score images with one or more quality models, then use the per-model sliders to decide
        how many low-quality images to discard. Flagged images move to
        <code>&lt;path&gt;/low_quality/</code> — non-destructive and reversible by moving them
        back. Building is incremental: re-running only scores new images.
      </p>
      <div class="qi-view__setup-row">
        <div class="qi-view__field qi-view__path">
          <label class="qi-view__label">Dataset folder</label>
          <PathFieldControl
            v-model="path"
            placeholder="e.g. /path/to/dataset/images"
            expect="dir"
            required
          />
        </div>
        <div class="qi-view__field qi-view__models-field">
          <label class="qi-view__label">Models</label>
          <el-select
            v-model="selectedModels"
            multiple
            placeholder="Select quality models"
            class="w-full"
          >
            <el-option
              v-for="m in MODEL_OPTIONS"
              :key="m.value"
              :label="m.label"
              :value="m.value"
            />
          </el-select>
        </div>
        <el-button
          type="primary"
          :loading="building"
          :disabled="!canBuild"
          @click="buildIndex"
        >
          {{ hasIndex ? "Refresh index" : "Build index" }}
        </el-button>
      </div>

      <div v-if="building" class="qi-view__build-progress">
        <el-progress
          :percentage="Math.min(100, Math.round(progress?.percent ?? 0))"
          striped
          :striped-flow="true"
          :duration="10"
        />
        <el-text size="small" type="info" class="qi-view__progress-msg">
          {{ progress?.msg ?? "Building quality index — models may download on first use…" }}
        </el-text>
      </div>

      <el-alert
        v-if="buildError"
        type="error"
        :title="buildError"
        :closable="false"
        show-icon
        class="mt-8"
      />
    </el-card>

    <el-card v-if="!hasIndex && !building" shadow="never" class="qi-view__placeholder">
      <el-empty
        description="Set a dataset folder and models, then click 'Build index' to score your images."
        :image-size="64"
      />
    </el-card>

    <el-card
      v-for="model in indexedModels"
      :key="model"
      shadow="never"
      class="qi-view__model-card"
    >
      <template #header>
        <div class="qi-view__card-header">
          <span class="qi-view__model-label">{{ modelLabel(model) }}</span>
          <el-tag size="small" type="info" effect="plain">{{ model }}</el-tag>
        </div>
      </template>

      <!-- Stats row -->
      <div v-if="getModelStats(model)" class="qi-view__stats-row">
        <el-descriptions :column="4" border size="small">
          <el-descriptions-item label="Reference">
            <el-tooltip
              content="Total scored images including those moved to low_quality/"
              :show-after="400"
            >
              <span>{{ getModelStats(model)!.reference }}</span>
            </el-tooltip>
          </el-descriptions-item>
          <el-descriptions-item label="Present">
            <el-tooltip content="Images still in the dataset folder" :show-after="400">
              <span>{{ getModelStats(model)!.present }}</span>
            </el-tooltip>
          </el-descriptions-item>
          <el-descriptions-item label="Min score">
            {{ getModelStats(model)!.min.toFixed(3) }}
          </el-descriptions-item>
          <el-descriptions-item label="Max score">
            {{ getModelStats(model)!.max.toFixed(3) }}
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- Percentile cull slider -->
      <div class="qi-view__slider-section">
        <div class="qi-view__slider-meta">
          <span class="qi-view__slider-heading">Discard lowest %</span>
          <el-text size="small" type="info">
            Drop the lowest N% of images by this model's score. The cutoff is computed over
            the full reference set, so re-running never erodes further. Multiple models use
            union: an image is dropped if ANY selected model flags it.
          </el-text>
        </div>
        <el-slider
          v-model="sliderValues[model]"
          :min="0"
          :max="100"
          :step="1"
          show-input
          class="qi-view__slider"
          @change="onSliderChange"
        />
      </div>

      <!-- Cull preview for this model -->
      <div v-if="cullPreview" class="qi-view__cull-preview">
        <el-text size="small">
          This model would flag
          <strong>{{ cullPreview.per_model[model] ?? 0 }}</strong> image(s).
          <template v-if="indexedModels.length > 1">
            Across all models (union):
            <strong>{{ cullPreview.union }}</strong> of
            <strong>{{ cullPreview.present }}</strong> images would be dropped.
          </template>
          <template v-else>
            <strong>{{ cullPreview.union }}</strong> of
            <strong>{{ cullPreview.present }}</strong> images would be dropped.
          </template>
        </el-text>
      </div>

      <!-- Sample size + gallery -->
      <div class="qi-view__gallery-controls">
        <span class="qi-view__gallery-size-label">Worst-image sample:</span>
        <el-radio-group
          v-model="sampleSizes[model]"
          size="small"
          @change="onSampleSizeChange(model)"
        >
          <el-radio-button :value="100">100</el-radio-button>
          <el-radio-button :value="200">200</el-radio-button>
          <el-radio-button :value="1000">1000</el-radio-button>
        </el-radio-group>
        <el-text size="small" type="info">
          Lowest-scoring images appear first — use them to calibrate the cut-off slider above.
        </el-text>
      </div>

      <template v-if="worstLoading[model]">
        <el-text size="small" type="info" class="qi-view__gallery-loading">Loading…</el-text>
      </template>
      <template v-else>
        <!-- Legend: only shown when a cutoff is active for this model -->
        <div
          v-if="sliderValues[model] > 0 && cullPreview && getWorstItems(model).length"
          class="qi-view__gallery-legend"
        >
          <span class="qi-view__legend-swatch qi-view__legend-swatch--removed" aria-hidden="true" />
          <el-text size="small" type="danger">Will be removed</el-text>
          <span class="qi-view__legend-swatch qi-view__legend-swatch--kept" aria-hidden="true" />
          <el-text size="small" type="info">Kept</el-text>
          <el-text size="small" type="info">
            — lowest {{ sliderValues[model] }}% by score
          </el-text>
        </div>

        <div v-if="getWorstItems(model).length" class="qi-view__gallery">
          <div
            v-for="item in getWorstItems(model)"
            :key="item.path"
            class="qi-view__thumb-cell"
            :class="{ 'qi-view__thumb-cell--removed': itemWillRemove(model, item) }"
          >
            <div class="qi-view__thumb-wrap">
              <img
                :src="api.datasetPreviewImageUrl(item.token)"
                class="qi-view__thumb"
                :alt="item.name"
                loading="lazy"
              />
              <!-- Red tint overlay for removed items -->
              <div
                v-if="itemWillRemove(model, item)"
                class="qi-view__thumb-overlay"
                aria-hidden="true"
              />
              <!-- Corner badge -->
              <span
                v-if="itemWillRemove(model, item)"
                class="qi-view__thumb-badge"
                aria-label="will be removed"
              >×</span>
            </div>
            <div
              class="qi-view__thumb-score"
              :class="{ 'qi-view__thumb-score--removed': itemWillRemove(model, item) }"
            >
              {{ item.quality.toFixed(1) }}
            </div>
            <div class="qi-view__thumb-name" :title="item.name">{{ item.name }}</div>
          </div>
        </div>
        <el-empty
          v-else
          description="No scored images found"
          :image-size="40"
          class="qi-view__gallery-empty"
        />
      </template>
    </el-card>

    <!-- Apply (one button, shown when at least one model has a non-zero slider) -->
    <div v-if="hasIndex && hasAnyCull" class="qi-view__apply-row">
      <el-button
        type="primary"
        size="large"
        :loading="applying"
        @click="openApplyConfirm"
      >
        Apply — move flagged images to low_quality/
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import { usePrepJobLive } from "../composables/usePrepJobLive";
import PathFieldControl from "../components/PathFieldControl.vue";
import type {
  QualityIndexStatsResult,
  QualityIndexWorstItem,
  QualityIndexCullPreviewResult,
} from "../types/api";

// ---------------------------------------------------------------------------
// Model options (value / label as specified)
// ---------------------------------------------------------------------------

const MODEL_OPTIONS = [
  { value: "aesthetic", label: "Aesthetic — anime booru appeal" },
  { value: "clipiqa", label: "CLIP-IQA — any domain" },
  { value: "arniqa", label: "ARNIQA — any domain" },
  { value: "musiq", label: "MUSIQ — photos" },
  { value: "maniqa", label: "MANIQA — photos" },
  { value: "brisque", label: "BRISQUE — classic" },
  { value: "niqe", label: "NIQE — classic" },
] as const;

function modelLabel(model: string): string {
  return MODEL_OPTIONS.find((m) => m.value === model)?.label ?? model;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const path = ref("");
const selectedModels = ref<string[]>([]);
const building = ref(false);
const buildError = ref("");
const applying = ref(false);
const jobId = ref<string | undefined>(undefined);

/** Models that have a completed index (reference > 0). */
const indexedModels = ref<string[]>([]);

/** Per-model stats; undefined = not yet loaded. */
const modelStatsMap = reactive<Record<string, QualityIndexStatsResult | undefined>>({});

/** Per-model worst items. */
const worstItemsMap = reactive<Record<string, QualityIndexWorstItem[]>>({});

/** Per-model loading flag for the worst-items gallery. */
const worstLoading = reactive<Record<string, boolean>>({});

/** Per-model sample sizes (100 / 200 / 1000). */
const sampleSizes = reactive<Record<string, number>>({});

/** Per-model cull percentile sliders (0–100, default 10). */
const sliderValues = reactive<Record<string, number>>({});

/** Live cull-preview result (reflects all sliders simultaneously). */
const cullPreview = ref<QualityIndexCullPreviewResult | null>(null);

// ---------------------------------------------------------------------------
// Derived state
// ---------------------------------------------------------------------------

const canBuild = computed(
  () => !!path.value.trim() && selectedModels.value.length > 0 && !building.value
);
const hasIndex = computed(() => indexedModels.value.length > 0);
const hasAnyCull = computed(() =>
  indexedModels.value.some((m) => (sliderValues[m] ?? 0) > 0)
);

// ---------------------------------------------------------------------------
// Accessors (keep template type-safe without non-null assertions in markup)
// ---------------------------------------------------------------------------

function getModelStats(model: string): QualityIndexStatsResult | null {
  return modelStatsMap[model] ?? null;
}

function getWorstItems(model: string): QualityIndexWorstItem[] {
  return worstItemsMap[model] ?? [];
}

/**
 * Returns true when the current cull-preview cutoff for `model` means this
 * item would be removed (quality strictly below the cutoff score).
 */
function itemWillRemove(model: string, item: QualityIndexWorstItem): boolean {
  const cutoff = cullPreview.value?.cutoffs?.[model];
  return cutoff != null && item.quality < cutoff;
}

// ---------------------------------------------------------------------------
// Live stream
// ---------------------------------------------------------------------------

let cullPreviewTimer: ReturnType<typeof setTimeout> | null = null;

const { progress } = usePrepJobLive(() => jobId.value, {
  onRunFinished: () => {
    void handleRunFinished();
  },
});

// ---------------------------------------------------------------------------
// Index build flow
// ---------------------------------------------------------------------------

async function buildIndex(): Promise<void> {
  if (!canBuild.value) return;
  building.value = true;
  buildError.value = "";
  try {
    const job = await api.createPrepJob({
      stage: "index",
      config: {
        path: path.value.trim(),
        index: { models: [...selectedModels.value] },
      },
      start_now: true,
    });
    jobId.value = String(job.id);
  } catch (e) {
    building.value = false;
    buildError.value = e instanceof Error ? e.message : String(e);
  }
}

async function handleRunFinished(): Promise<void> {
  const id = jobId.value;
  try {
    if (!id) return;
    const { report } = await api.prepJobReport(id);
    if (!report) return;

    const stats = (report as Record<string, unknown>).stats as
      | Record<string, { reference: number; present: number; min: number; max: number }>
      | undefined;

    const newModels: string[] = stats
      ? Object.keys(stats).filter((m) => stats[m].reference > 0)
      : [];

    // Initialize per-model defaults before rendering the cards
    for (const model of newModels) {
      if (sliderValues[model] == null) sliderValues[model] = 10;
      if (sampleSizes[model] == null) sampleSizes[model] = 100;
      worstItemsMap[model] = [];
    }

    indexedModels.value = newModels;

    // Load live stats and worst items in parallel
    await Promise.all([
      ...newModels.map((m) => loadModelStats(m)),
      ...newModels.map((m) => loadWorst(m)),
    ]);

    await fetchCullPreview();
  } catch (e) {
    buildError.value = e instanceof Error ? e.message : String(e);
  } finally {
    building.value = false;
    jobId.value = undefined;
  }
}

// ---------------------------------------------------------------------------
// Data loaders
// ---------------------------------------------------------------------------

async function loadModelStats(model: string): Promise<void> {
  try {
    modelStatsMap[model] = await api.qualityIndexStats(path.value, model);
  } catch {
    // Leave absent — stats section will not render
  }
}

async function loadWorst(model: string): Promise<void> {
  worstLoading[model] = true;
  try {
    const { items } = await api.qualityIndexWorst({
      path: path.value,
      model,
      limit: sampleSizes[model] ?? 100,
    });
    worstItemsMap[model] = items;
  } catch {
    worstItemsMap[model] = [];
  } finally {
    worstLoading[model] = false;
  }
}

async function fetchCullPreview(): Promise<void> {
  if (!path.value || indexedModels.value.length === 0) return;
  const perModel: Record<string, number> = {};
  for (const model of indexedModels.value) {
    perModel[model] = sliderValues[model] ?? 0;
  }
  try {
    cullPreview.value = await api.qualityIndexCullPreview({
      path: path.value,
      per_model: perModel,
    });
  } catch {
    // Best-effort; don't surface preview errors
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

function onSliderChange(): void {
  if (cullPreviewTimer) clearTimeout(cullPreviewTimer);
  cullPreviewTimer = setTimeout(() => {
    void fetchCullPreview();
  }, 250);
}

function onSampleSizeChange(model: string): void {
  void loadWorst(model);
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

async function openApplyConfirm(): Promise<void> {
  const modelsWithCull = indexedModels.value.filter((m) => (sliderValues[m] ?? 0) > 0);
  const unionCount = cullPreview.value?.union ?? "?";
  const present = cullPreview.value?.present ?? "?";
  const modelSummary = modelsWithCull.map((m) => `${m} (${sliderValues[m]}%)`).join(", ");

  try {
    await ElMessageBox.confirm(
      `Move ${unionCount} of ${present} images to ${path.value}/low_quality/?\n` +
        `Models: ${modelSummary}.\n\n` +
        `This is non-destructive — restore images by moving them back from low_quality/.`,
      "Apply quality cull",
      {
        confirmButtonText: "Move images",
        cancelButtonText: "Cancel",
        type: "warning",
      }
    );
  } catch {
    return; // User cancelled
  }

  applying.value = true;
  try {
    const perModel: Record<string, number> = {};
    for (const model of indexedModels.value) {
      const pct = sliderValues[model] ?? 0;
      if (pct > 0) perModel[model] = pct;
    }
    const result = await api.qualityIndexApply({ path: path.value, per_model: perModel });
    ElMessage.success(`Moved ${result.moved} image(s) to low_quality/`);

    // Refresh stats + worst items so moved images leave the present set
    await Promise.all([
      ...indexedModels.value.map((m) => loadModelStats(m)),
      ...indexedModels.value.map((m) => loadWorst(m)),
    ]);
    await fetchCullPreview();
  } catch (e) {
    ElMessage.error(e instanceof Error ? e.message : String(e));
  } finally {
    applying.value = false;
  }
}
</script>

<style scoped>
.qi-view {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.qi-view__head {
  justify-content: flex-start;
  align-items: center;
  gap: var(--rf-space-sm);
  margin-bottom: 0;
}

.qi-view__title {
  font-size: 16px;
  font-weight: 600;
}

.qi-view__setup-row {
  display: flex;
  gap: var(--rf-space-xs);
  align-items: flex-end;
  flex-wrap: wrap;
}

.qi-view__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.qi-view__label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.qi-view__path {
  flex: 1;
  min-width: 240px;
}

.qi-view__models-field {
  flex: 1;
  min-width: 280px;
}

.qi-view__build-progress {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.qi-view__progress-msg {
  display: block;
}

.qi-view__placeholder :deep(.el-card__body) {
  padding: var(--rf-space-md);
}

.qi-view__card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.qi-view__model-label {
  font-weight: 500;
  flex: 1;
}

.qi-view__stats-row {
  margin-bottom: 14px;
}

.qi-view__slider-section {
  margin-bottom: 10px;
}

.qi-view__slider-meta {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 8px;
}

.qi-view__slider-heading {
  font-size: 13px;
  font-weight: 500;
}

.qi-view__slider {
  max-width: 600px;
}

.qi-view__cull-preview {
  margin-bottom: 12px;
  padding: 8px 10px;
  background: var(--el-fill-color-light);
  border-radius: var(--el-border-radius-base);
}

.qi-view__gallery-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}

.qi-view__gallery-size-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
}

.qi-view__gallery-loading {
  display: block;
  padding: 8px 0;
}

.qi-view__gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
  max-height: 480px;
  overflow: auto;
}

/* Gallery legend */
.qi-view__gallery-legend {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.qi-view__legend-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 3px;
  flex-shrink: 0;
}

.qi-view__legend-swatch--removed {
  background: var(--el-color-danger);
}

.qi-view__legend-swatch--kept {
  background: var(--el-fill-color-darker);
  border: 1px solid var(--el-border-color);
}

/* Thumbnail cells */
.qi-view__thumb-cell {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

/* Image wrapper: needed for overlay + badge positioning */
.qi-view__thumb-wrap {
  position: relative;
  border-radius: 5px;
  overflow: hidden;
  line-height: 0;
}

.qi-view__thumb {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: 5px;
  background: var(--el-fill-color-darker);
  display: block;
  transition: opacity 0.15s, filter 0.15s;
}

/* Removed-state visuals */
.qi-view__thumb-cell--removed .qi-view__thumb {
  opacity: 0.5;
  filter: saturate(0.3);
}

.qi-view__thumb-cell--removed .qi-view__thumb-wrap {
  outline: 2px solid var(--el-color-danger);
  outline-offset: -2px;
}

/* Semi-transparent red tint over the image */
.qi-view__thumb-overlay {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--el-color-danger) 20%, transparent);
  pointer-events: none;
}

/* Corner badge "×" */
.qi-view__thumb-badge {
  position: absolute;
  top: 4px;
  left: 4px;
  background: var(--el-color-danger);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  padding: 2px 4px;
  border-radius: 3px;
  pointer-events: none;
  user-select: none;
}

.qi-view__thumb-score {
  font-size: 11px;
  font-weight: 600;
  text-align: center;
  color: var(--el-text-color-primary);
}

.qi-view__thumb-score--removed {
  color: var(--el-color-danger);
}

.qi-view__thumb-name {
  font-size: 10px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: center;
}

.qi-view__gallery-empty :deep(.el-empty) {
  padding: 16px 0;
}

.qi-view__apply-row {
  display: flex;
  justify-content: flex-end;
  padding: 4px 0 8px;
}

.w-full {
  width: 100%;
}

.mt-8 {
  margin-top: 8px;
}

.page-hint {
  margin: 0 0 14px;
  color: var(--el-text-color-secondary);
  font-size: 13px;
  line-height: 1.5;
}
</style>

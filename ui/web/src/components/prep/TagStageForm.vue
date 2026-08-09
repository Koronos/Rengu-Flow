<!--
  Tagging options, extracted verbatim from PrepJobFormView.vue. Owns its own
  registry fetch (`api.prepModels("tag")`), the downloaded-models preselect and
  the per-model confidence floors.

  The `#extra` slot is the fourth cell of the switch grid; the view puts the
  "chain a caption job" toggle there (page-level behaviour, not a config field).
-->
<template>
  <h3 class="section-title">Tagging options</h3>
  <el-form label-position="top" :disabled="disabled">
    <el-form-item>
      <template #label>
        Models <FieldHelpIcon :field="help('Runs each model in sequence and merges per-image probabilities by max, so the ensemble catches tags any single model misses. Add a second model when one alone keeps missing specific tag categories.')" />
        <FieldPathTag path="tag.models" />
      </template>
      <el-text v-if="modelsLoading" size="small" type="info">Loading models…</el-text>
      <el-select
        v-else
        v-model="model.models"
        multiple
        filterable
        placeholder="Select tagger models"
        class="w-full"
      >
        <el-option
          v-for="m in tagModels"
          :key="m.id"
          :label="`${m.id}${m.downloaded ? ' ✓' : ' (will download)'}`"
          :value="m.id"
        >
          <span>{{ m.id }}</span>
          <el-tag v-if="m.downloaded" size="small" type="success" effect="plain" class="ml-8">downloaded</el-tag>
          <el-tag v-else size="small" type="warning" effect="plain" class="ml-8">will download</el-tag>
        </el-option>
      </el-select>
    </el-form-item>

    <!-- Per-model confidence floors: each model keeps its own, seeded from its
         defaults. 0 drops Character/Rating for that model; General is always kept. -->
    <h4 class="section-subtitle">
      Confidence per model <FieldHelpIcon :field="help('Each selected model has its own confidence floors, pre-filled with that model\'s own defaults. Set Character or Rating to 0 to drop that category for that model; general tags are always kept. Higher = fewer but surer tags.')" />
    </h4>
    <el-text v-if="!model.models.length" size="small" type="info" class="hint-text">
      Select at least one model above to set its confidence.
    </el-text>
    <div
      v-for="mid in model.models"
      v-else
      :key="mid"
      class="model-thresholds"
    >
      <span class="model-thresholds__name">{{ mid }}</span>
      <div class="model-thresholds__fields">
        <div class="conf-field">
          <label class="conf-label">General</label>
          <el-input-number
            v-model="thresholds[mid].general"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            class="conf-input"
          />
        </div>
        <div class="conf-field">
          <label class="conf-label">Character <span class="conf-off">0 = off</span></label>
          <el-input-number
            v-model="thresholds[mid].character"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            class="conf-input"
          />
        </div>
        <div class="conf-field">
          <label class="conf-label">Rating <span class="conf-off">0 = off</span></label>
          <el-input-number
            v-model="thresholds[mid].rating"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            class="conf-input"
          />
        </div>
      </div>
    </div>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Exclude tags <FieldHelpIcon :field="help('Strips these tags from every image output regardless of model confidence. Use when specific tags keep appearing that are wrong for your dataset style (e.g. realistic on anime images) or would bias training negatively.')" />
          <FieldPathTag path="tag.exclude_tags" />
        </template>
        <el-input-tag
          v-model="model.exclude_tags"
          clearable
          delimiter=","
          placeholder="e.g. realistic, 3d"
          class="w-full"
        />
      </el-form-item>
      <el-form-item>
        <template #label>
          Prepend tags <FieldHelpIcon :field="help('Inserts these tags at the start of every image\'s tag line before the tagger output. Use for your trigger word or any tag the model consistently misses.')" />
          <FieldPathTag path="tag.prepend_tags" />
        </template>
        <el-input-tag
          v-model="model.prepend_tags"
          clearable
          delimiter=","
          placeholder="e.g. my_trigger_word"
          class="w-full"
        />
      </el-form-item>
    </div>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Max tags <FieldHelpIcon :field="help('Hard cap on tags kept per image, highest-confidence first (default 40). Raise it if useful tags are being dropped; lower it to trim the low-confidence tail.')" />
          <FieldPathTag path="tag.max_tags" />
        </template>
        <el-input-number v-model="model.max_tags" :min="1" :max="500" placeholder="40" controls-position="right" class="w-full" />
      </el-form-item>
      <el-form-item>
        <template #label>
          Batch size <FieldHelpIcon :field="help('Images per ONNX forward pass (default 8). Raise it to tag faster on a card with spare VRAM; lower it if the job runs out of memory.')" />
          <FieldPathTag path="tag.batch_size" />
        </template>
        <el-input-number v-model="model.batch_size" :min="1" :max="64" placeholder="8" controls-position="right" class="w-full" />
      </el-form-item>
    </div>
    <el-form-item>
      <template #label>
        Target line <FieldHelpIcon :field="help('1-based caption line the tags are written to (default 1 = the tag line). Raise it to keep tags on a separate line, e.g. alongside a natural-language caption on line 2. The skip-when-not-overwriting check looks at this line.')" />
        <FieldPathTag path="tag.target_line" />
      </template>
      <el-input-number v-model="model.target_line" :min="1" :max="10" placeholder="1" controls-position="right" />
    </el-form-item>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Overwrite <FieldHelpIcon :field="help('Re-tags images that already have a tag line on line 1, replacing it. Turn on when you are changing models or thresholds and want to regenerate tags for the whole folder from scratch.')" />
          <FieldPathTag path="tag.overwrite" />
        </template>
        <el-switch v-model="model.overwrite" />
        <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
      </el-form-item>
      <el-form-item>
        <template #label>
          Quality tags <FieldHelpIcon :field="help('Runs the deepghs aesthetic model and prepends a booru quality tag (masterpiece … worst quality) to each caption, the anime-training convention. Adds a GPU pass and downloads the model on first use.')" />
          <FieldPathTag path="tag.quality_tags" />
        </template>
        <el-switch v-model="model.quality_tags" />
        <el-text class="ml-8" size="small">Prepend quality tag to each caption</el-text>
      </el-form-item>
      <el-form-item>
        <template #label>
          Underscores <FieldHelpIcon :field="help('Tag form: on keeps the original danbooru form (long_hair), off writes spaces (long hair). SDXL is usually trained with underscores; Cosmos is spaces-only. Tag-dropout control lists (undroppable tags) match both forms either way, so the choice only affects the written captions.')" />
          <FieldPathTag path="tag.underscores" />
        </template>
        <el-switch v-model="model.underscores" />
        <el-text class="ml-8" size="small">Keep original danbooru form (long_hair)</el-text>
      </el-form-item>
      <slot name="extra" />
    </div>
  </el-form>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import type { PropType } from "vue";
import { api } from "../../api";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import { copyKnown, help } from "./formHelpers";
import { modelThresholdDefaults } from "../../lib/prepStageConfig";
import type { ModelThresholds, PrepTagForm } from "../../lib/prepStageConfig";
import type { PrepModelInfo, PrepTagConfig } from "../../types/api";

const model = defineModel<PrepTagForm>({ required: true });
const thresholds = defineModel<Record<string, ModelThresholds>>("thresholds", {
  default: () => ({}),
});

const props = defineProps({
  /** `tag` section of a cloned job config; applied once the registry has loaded. */
  seed: { type: Object as PropType<PrepTagConfig | null>, default: null },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the whole stage form — the confidence inputs included.
   */
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits<{
  /** The tagger registry, so the parent can resolve per-model defaults too. */
  (e: "models-loaded", models: PrepModelInfo[]): void;
}>();

const tagModels = ref<PrepModelInfo[]>([]);
const modelsLoading = ref(false);

function modelDefaults(modelId: string): ModelThresholds {
  return modelThresholdDefaults(tagModels.value.find((x) => x.id === modelId));
}

/** Ensure every selected model has a thresholds row, seeded from its defaults. */
function syncThresholds(): void {
  for (const id of model.value.models) {
    if (!thresholds.value[id]) thresholds.value[id] = modelDefaults(id);
  }
}

watch(() => model.value.models.slice(), syncThresholds, { deep: true });

const loaded = ref(false);
let seedApplied = false;

/**
 * Seed from a cloned job config. Rebuilds the per-model thresholds from the
 * stored overrides (inverse of `buildStageConfig`): include_*=false -> 0 (off);
 * otherwise the stored *_threshold.
 */
function applySeed(): void {
  const seed = props.seed;
  if (!seed || seedApplied) return;
  seedApplied = true;
  copyKnown(model.value as unknown as Record<string, unknown>, seed);
  syncThresholds();
  for (const [mid, o] of Object.entries(seed.overrides ?? {})) {
    const t = thresholds.value[mid] ?? modelDefaults(mid);
    if (typeof o.general_threshold === "number") t.general = o.general_threshold;
    t.character =
      o.include_character === false
        ? 0
        : typeof o.character_threshold === "number"
          ? o.character_threshold
          : t.character;
    t.rating =
      o.include_rating === false
        ? 0
        : typeof o.rating_threshold === "number"
          ? o.rating_threshold
          : t.rating;
    thresholds.value[mid] = t;
  }
}

async function loadModels(): Promise<void> {
  modelsLoading.value = true;
  try {
    const res = await api.prepModels("tag");
    tagModels.value = res.models || [];
    emit("models-loaded", tagModels.value);
    // The preselect fills a GAP, it never replaces a choice: `model` is a shared v-model, and in
    // the workflow drawer a write here is an edit the parent saves. Overwriting a seeded
    // selection would silently persist the registry's guess over the user's saved one.
    if (!model.value.models.length) {
      // pre-select downloaded models; fall back to the registry's default ensemble
      model.value.models = tagModels.value.filter((m) => m.downloaded).map((m) => m.id);
      if (!model.value.models.length) {
        model.value.models = tagModels.value.slice(0, 2).map((m) => m.id);
      }
    }
    syncThresholds();
  } catch {
    // models endpoint may not be implemented yet — silently degrade
  } finally {
    modelsLoading.value = false;
    loaded.value = true;
    // A seed that arrived while the registry was in flight is applied now (the watcher below
    // ignores it until `loaded`); `applySeed` is one-shot, so a seed already applied is a no-op.
    applySeed();
  }
}

// A clone seed overrides the preselect, whichever of the two settles last.
watch(
  () => props.seed,
  () => {
    if (loaded.value) applySeed();
  }
);

/**
 * Seed first, and **synchronously**.
 *
 * `loadModels` awaits, and Vue flushes watchers in the microtask that await opens — so seeding
 * after it means the parent sees one tick of preselect-shaped state and takes it for an edit.
 * In the workflow drawer that tick reaches `watch(builtConfig)`, which emits `update:node`, and
 * merely *opening* a saved step autosaves the registry's guess over its config.
 */
onMounted(() => {
  applySeed();
  void loadModels();
});
</script>

<style scoped>
.section-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.section-subtitle {
  margin: 0 0 var(--rf-space-xs);
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.form-row-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 600px) {
  .form-row-2 {
    grid-template-columns: 1fr;
  }
}
.hint-text {
  display: block;
  margin-top: 4px;
  margin-bottom: 8px;
}
.ml-8 {
  margin-left: 8px;
}
.model-thresholds {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--rf-space-sm) var(--rf-space-md);
  padding: var(--rf-space-sm) var(--rf-space-md);
  margin-bottom: var(--rf-space-xs);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  background: var(--el-fill-color-blank);
}
.model-thresholds__name {
  font-family: var(--rf-font-mono);
  font-size: 13px;
  font-weight: 600;
  flex: 0 0 auto;
  min-width: 140px;
}
.model-thresholds__fields {
  display: flex;
  flex-wrap: wrap;
  gap: var(--rf-space-md);
}
.conf-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.conf-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.conf-off {
  color: var(--el-text-color-placeholder);
}
.conf-input {
  width: 130px;
}
</style>

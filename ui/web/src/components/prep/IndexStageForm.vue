<!--
  Quality-index options. NEW construction, not an extraction: the `index` stage
  had no form at all — `buildConfig()` had no branch for it and fell through to
  `clean`, so `/prep/new/index` could not produce a valid job.

  `IndexStageConfig` (rengu_flow/prep/config.py) is a single field: the model ids
  to score with. "aesthetic" is the deepghs booru-appeal model; every other id is
  passed straight to pyiqa, so free-form entries are allowed.
-->
<template>
  <h3 class="section-title">Quality index options</h3>
  <el-alert
    type="info"
    :closable="false"
    show-icon
    class="mt-8 mb-12"
    title="Scores every image once and stores the results, so culling later never pays the model again. Only images that are missing or have changed get scored — re-running is cheap."
  />
  <el-form label-position="top" :disabled="disabled">
    <el-form-item required>
      <template #label>
        Quality models <FieldHelpIcon :field="help('Which scorers to index with. Each model keeps its own scores and the cull unions them, so adding a second model catches images the first one rates fine. Aesthetic is the anime booru-appeal model; the rest are pyiqa no-reference models — any other pyiqa model id can be typed in.')" />
        <FieldPathTag path="index.models" />
      </template>
      <el-select
        v-model="model.models"
        multiple
        filterable
        allow-create
        default-first-option
        placeholder="Select one or more quality models"
        class="w-full"
      >
        <el-option
          v-for="m in MODEL_OPTIONS"
          :key="m.value"
          :label="m.label"
          :value="m.value"
        />
      </el-select>
      <el-text v-if="!model.models.length" size="small" type="info" class="hint-text">
        Pick at least one model — the job fails without one.
      </el-text>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { onMounted, watch } from "vue";
import type { PropType } from "vue";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import { copyKnown, help } from "./formHelpers";
import type { PrepIndexForm } from "../../lib/prepStageConfig";
import type { PrepIndexConfig } from "../../types/api";

/** Same catalogue QualityIndexView offers, kept in the same order. */
const MODEL_OPTIONS = [
  { value: "aesthetic", label: "Aesthetic — anime booru appeal" },
  { value: "clipiqa", label: "CLIP-IQA — any domain" },
  { value: "arniqa", label: "ARNIQA — any domain" },
  { value: "musiq", label: "MUSIQ — photos" },
  { value: "maniqa", label: "MANIQA — photos" },
  { value: "brisque", label: "BRISQUE — classic" },
  { value: "niqe", label: "NIQE — classic" },
];

const model = defineModel<PrepIndexForm>({ required: true });

const props = defineProps({
  /** `index` section of a cloned job config. */
  seed: { type: Object as PropType<PrepIndexConfig | null>, default: null },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the whole stage form.
   */
  disabled: { type: Boolean, default: false },
});

let seedApplied = false;

function applySeed(): void {
  if (!props.seed || seedApplied) return;
  seedApplied = true;
  copyKnown(model.value as unknown as Record<string, unknown>, props.seed);
}

watch(() => props.seed, applySeed);
onMounted(applySeed);
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
.mt-8 {
  margin-top: 8px;
}
</style>

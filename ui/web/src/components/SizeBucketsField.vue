<template>
  <div class="size-buckets-field">
    <template v-if="!showJson">
      <el-empty v-if="!rows.length" description="No size buckets yet" :image-size="56">
        <el-button type="primary" :icon="Plus" @click="addRow">Add bucket</el-button>
      </el-empty>
      <div v-for="(row, idx) in rows" :key="idx" class="bucket-row">
        <el-input-number
          :model-value="row[0]"
          :min="1"
          :step="64"
          controls-position="right"
          class="field-narrow bucket-input"
          placeholder="Width"
          @update:model-value="(v) => updateRow(idx, 0, v)"
        />
        <span class="dim-sep">×</span>
        <el-input-number
          :model-value="row[1]"
          :min="1"
          :step="64"
          controls-position="right"
          class="field-narrow bucket-input"
          placeholder="Height"
          @update:model-value="(v) => updateRow(idx, 1, v)"
        />
        <el-input-number
          :model-value="row[2]"
          :min="1"
          :step="1"
          controls-position="right"
          class="field-narrow bucket-input"
          placeholder="Frames"
          @update:model-value="(v) => updateRow(idx, 2, v)"
        />
        <el-text type="info" size="small" class="frames-label">frames</el-text>
        <el-button type="danger" link :icon="Delete" @click="removeRow(idx)" />
      </div>

      <el-space v-if="rows.length" wrap class="actions-row">
        <el-button size="small" @click="addRow">Add bucket</el-button>
        <el-button size="small" link @click="showJson = true">Edit as JSON</el-button>
      </el-space>

      <div v-if="presetOptions.length" class="preset-row">
        <el-text type="info" size="small" class="preset-label">Quick add</el-text>
        <el-button
          v-for="preset in presetOptions"
          :key="sizeBucketKey(preset)"
          size="small"
          round
          @click="addPreset(preset)"
        >
          {{ formatSizeBucketLabel(preset) }}
        </el-button>
      </div>
    </template>

    <JsonFallbackEditor
      v-else
      :json-text="jsonText"
      :error="jsonError"
      :can-return="canUseTableEditor"
      :rows="5"
      placeholder="[[512, 512, 1], [768, 768, 1]]"
      back-label="Back to table editor"
      @update:json="onJsonInput"
      @return="showJson = false"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, type PropType } from "vue";
import { Delete, Plus } from "@element-plus/icons-vue";
import JsonFallbackEditor from "./JsonFallbackEditor.vue";
import { useJsonFieldToggle } from "../composables/useJsonFieldToggle";
import {
  SIZE_BUCKET_PRESETS,
  formatSizeBucketLabel,
  parseSizeBuckets,
  sizeBucketKey,
  sizeBucketsFormValue,
  sizeBucketsNeedJsonEditor,
  validateSizeBucketsJson,
  type SizeBucket,
} from "../lib/sizeBuckets";

const props = defineProps({
  modelValue: {
    type: [Array, String] as PropType<SizeBucket[] | string | null>,
    default: () => [],
  },
});

const emit = defineEmits(["update:modelValue"]);

const { showJson, jsonText, jsonError, canUseTableEditor, onJsonInput } =
  useJsonFieldToggle({
    modelValue: () => props.modelValue,
    needsJsonEditor: sizeBucketsNeedJsonEditor,
    validate: validateSizeBucketsJson,
    emit: (v) => emit("update:modelValue", v),
  });

const rows = computed(() => parseSizeBuckets(props.modelValue));

const presetOptions = computed(() => {
  const current = new Set(rows.value.map(sizeBucketKey));
  return SIZE_BUCKET_PRESETS.filter((preset) => !current.has(sizeBucketKey(preset)));
});

function emitRows(next: SizeBucket[]): void {
  emit("update:modelValue", sizeBucketsFormValue(next));
}

function addRow(): void {
  emitRows([...rows.value, [512, 512, 1]]);
}

function removeRow(index: number): void {
  const next = rows.value.filter((_, i) => i !== index);
  emitRows(next);
}

function updateRow(index: number, dim: 0 | 1 | 2, value: number | undefined): void {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return;
  const next = rows.value.map((row, i) => {
    if (i !== index) return row;
    const copy: SizeBucket = [...row];
    copy[dim] = Math.round(n);
    return copy;
  });
  emitRows(next);
}

function addPreset(preset: SizeBucket): void {
  emitRows([...rows.value, preset]);
}
</script>

<style scoped>
.size-buckets-field {
  width: 100%;
}
.bucket-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.dim-sep {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.frames-label {
  flex-shrink: 0;
}
.actions-row {
  margin-top: 4px;
}
.preset-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}
.preset-label {
  margin-right: 2px;
}
</style>

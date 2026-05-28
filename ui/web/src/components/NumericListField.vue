<template>
  <div class="numeric-list">
    <el-input-tag
      :model-value="tagValues"
      clearable
      delimiter=","
      :placeholder="placeholder"
      class="field-full"
      @update:model-value="onTagsChange"
    />
    <div v-if="quickAddOptions.length" class="preset-row">
      <el-text type="info" size="small" class="preset-label">Quick add</el-text>
      <el-button
        v-for="opt in quickAddOptions"
        :key="opt"
        size="small"
        round
        @click="addValue(opt)"
      >
        {{ opt }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { parseIntegerList } from "../lib/integerList";
import { parseNumberList } from "../lib/numberList";
import type { PropType } from "vue";
import type { RawListInput } from "../types/forms";

const props = defineProps({
  modelValue: { type: [Array, String, Number] as PropType<RawListInput>, default: () => [] },
  presetOptions: { type: Array as PropType<Array<string | number>>, default: () => [] },
  placeholder: { type: String, default: "Type a number and press Enter" },
  /** When true, values are parsed as integers (integer_list fields). */
  integer: { type: Boolean, default: false },
  min: { type: Number, default: undefined },
  max: { type: Number, default: undefined },
});

const emit = defineEmits(["update:modelValue"]);

const effectiveMin = computed(() => {
  if (props.min !== undefined) return props.min;
  return props.integer ? 1 : undefined;
});

function parse(raw: RawListInput): number[] {
  return props.integer ? parseIntegerList(raw) : parseNumberList(raw);
}

function clamp(nums: number[]): number[] {
  let filtered = nums;
  const min = effectiveMin.value;
  const max = props.max;
  if (min !== undefined) {
    filtered = filtered.filter((n) => n >= min);
  }
  if (max !== undefined) {
    filtered = filtered.filter((n) => n <= max);
  }
  return filtered;
}

const sortedValues = computed(() => clamp(parse(props.modelValue)));

const tagValues = computed(() => sortedValues.value.map(String));

const quickAddOptions = computed(() => {
  const min = effectiveMin.value ?? -Infinity;
  const fromSchema = (props.presetOptions || [])
    .map((o) =>
      props.integer
        ? Number.parseInt(String(o), 10)
        : Number.parseFloat(String(o))
    )
    .filter((n) => Number.isFinite(n) && n >= min);
  const current = new Set(sortedValues.value);
  return [...new Set(fromSchema)]
    .filter((n) => !current.has(n))
    .sort((a, b) => a - b)
    .slice(0, 8);
});

function emitNumbers(nums: number[]): void {
  const filtered = clamp(nums);
  emit("update:modelValue", filtered.length ? filtered : "");
}

function onTagsChange(raw?: string[]): void {
  emitNumbers(parse(raw ?? []));
}

function addValue(n: number): void {
  const merged = [...new Set([...sortedValues.value, n])].sort((a, b) => a - b);
  emitNumbers(merged);
}
</script>

<style scoped>
.field-full {
  width: 100%;
}
.preset-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
}
.preset-label {
  margin-right: 2px;
}
</style>

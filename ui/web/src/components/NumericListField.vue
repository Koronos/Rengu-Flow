<template>
  <div class="numeric-list">
    <el-input-tag
      :model-value="tagValues"
      clearable
      delimiter=","
      :placeholder="placeholder"
      class="field-full"
      @update:model-value="onTagsChange"
      @blur="onBlurTrim"
    />
    <div v-if="quickAddOptions.length" class="preset-row">
      <el-tooltip
        content="Insert a common preset value into the list above"
        placement="top"
      >
        <el-text type="info" size="small" class="preset-label">Quick add</el-text>
      </el-tooltip>
      <el-button
        v-for="opt in quickAddOptions"
        :key="opt"
        size="small"
        round
        @click="addValue(opt)"
      >
        {{ formatPresetLabel(opt) }}
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { formatDefaultNumber } from "../lib/defaultFormat";
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
  maxLength: { type: Number, default: undefined },
});

const emit = defineEmits(["update:modelValue"]);

const effectiveMin = computed(() => {
  if (props.min !== undefined) return props.min;
  return props.integer ? 1 : undefined;
});

function parse(raw: RawListInput): number[] {
  if (props.integer) return parseIntegerList(raw);
  return parseNumberList(raw, props.maxLength);
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

const atMaxLength = computed(
  () => props.maxLength !== undefined && sortedValues.value.length >= props.maxLength
);

const quickAddOptions = computed(() => {
  if (atMaxLength.value) return [];
  const min = effectiveMin.value ?? -Infinity;
  const fromSchema = (props.presetOptions || [])
    .map((o) =>
      props.integer
        ? Number.parseInt(String(o), 10)
        : Number.parseFloat(String(o))
    )
    .filter((n) => Number.isFinite(n) && n >= min);
  const current = new Set(sortedValues.value);
  const available = [...new Set(fromSchema)].filter((n) => !current.has(n));
  if (props.maxLength !== undefined) {
    const slots = props.maxLength - sortedValues.value.length;
    return available.slice(0, Math.max(0, slots));
  }
  return available.sort((a, b) => a - b).slice(0, 8);
});

function emitNumbers(nums: number[] | RawListInput): void {
  const filtered = clamp(parse(nums));
  emit("update:modelValue", filtered.length ? filtered : "");
}

function onTagsChange(raw?: string[]): void {
  emitNumbers(parse(raw ?? []));
}

function formatPresetLabel(opt: number): string {
  return props.integer ? String(opt) : formatDefaultNumber(opt);
}

function addValue(n: number): void {
  if (atMaxLength.value) return;
  const current = sortedValues.value;
  if (current.includes(n)) return;
  emitNumbers([...current, n]);
}

function onBlurTrim(): void {
  emitNumbers(props.modelValue);
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

<template>
  <div class="number-list">
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
import { parseNumberList } from "../lib/numberList";

const props = defineProps({
  modelValue: { type: [Array, String, Number], default: () => [] },
  presetOptions: { type: Array, default: () => [] },
  placeholder: { type: String, default: "Type a number and press Enter" },
  min: { type: Number, default: undefined },
  max: { type: Number, default: undefined },
});

const emit = defineEmits(["update:modelValue"]);

const sortedValues = computed(() => {
  let nums = parseNumberList(props.modelValue);
  const min = props.min;
  const max = props.max;
  if (min !== undefined) {
    nums = nums.filter((n) => n >= min);
  }
  if (max !== undefined) {
    nums = nums.filter((n) => n <= max);
  }
  return nums;
});

const tagValues = computed(() => sortedValues.value.map(String));

const quickAddOptions = computed(() => {
  const fromSchema = (props.presetOptions || [])
    .map((o) => Number.parseFloat(String(o)))
    .filter((n) => Number.isFinite(n));
  const current = new Set(sortedValues.value);
  return [...new Set(fromSchema)]
    .filter((n) => !current.has(n))
    .sort((a, b) => a - b)
    .slice(0, 8);
});

function emitNumbers(nums) {
  let filtered = nums;
  const min = props.min;
  const max = props.max;
  if (min !== undefined) {
    filtered = filtered.filter((n) => n >= min);
  }
  if (max !== undefined) {
    filtered = filtered.filter((n) => n <= max);
  }
  emit("update:modelValue", filtered.length ? filtered : "");
}

function onTagsChange(raw) {
  emitNumbers(parseNumberList(raw));
}

function addValue(n) {
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

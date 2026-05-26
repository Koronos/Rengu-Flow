<template>
  <div class="integer-list">
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

<script setup>
import { computed } from "vue";
import { parseIntegerList } from "../lib/integerList";

const props = defineProps({
  modelValue: { type: [Array, String, Number], default: () => [] },
  presetOptions: { type: Array, default: () => [] },
  placeholder: { type: String, default: "Type a number and press Enter" },
  min: { type: Number, default: 1 },
});

const emit = defineEmits(["update:modelValue"]);

const sortedValues = computed(() => {
  const nums = parseIntegerList(props.modelValue);
  return nums.filter((n) => n >= props.min);
});

const tagValues = computed(() => sortedValues.value.map(String));

const quickAddOptions = computed(() => {
  const fromSchema = (props.presetOptions || [])
    .map((o) => Number.parseInt(String(o), 10))
    .filter((n) => Number.isFinite(n) && n >= props.min);
  const current = new Set(sortedValues.value);
  return [...new Set(fromSchema)]
    .filter((n) => !current.has(n))
    .sort((a, b) => a - b)
    .slice(0, 8);
});

function emitNumbers(nums) {
  const filtered = nums.filter((n) => n >= props.min);
  emit("update:modelValue", filtered.length ? filtered : "");
}

function onTagsChange(raw) {
  emitNumbers(parseIntegerList(raw));
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

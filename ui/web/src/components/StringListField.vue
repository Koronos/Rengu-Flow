<template>
  <div class="string-list">
    <el-select
      v-if="hasPresetOptions"
      :model-value="tagValues"
      multiple
      filterable
      allow-create
      default-first-option
      collapse-tags
      collapse-tags-tooltip
      :max-collapse-tags="3"
      :placeholder="placeholder"
      class="field-full"
      @update:model-value="onSelectChange"
    >
      <el-option
        v-for="opt in presetOptions"
        :key="opt"
        :label="opt"
        :value="opt"
      />
    </el-select>
    <el-input-tag
      v-else
      :model-value="tagValues"
      clearable
      delimiter=","
      :placeholder="placeholder"
      class="field-full"
      @update:model-value="onSelectChange"
    />
    <el-text v-if="hint" type="info" size="small" class="list-hint">{{ hint }}</el-text>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { parseStringList } from "../lib/stringList";
import type { PropType } from "vue";
import type { RawListInput } from "../types/forms";

const props = defineProps({
  modelValue: { type: [Array, String] as PropType<RawListInput>, default: () => [] },
  presetOptions: { type: Array as PropType<Array<string | number>>, default: () => [] },
  placeholder: { type: String, default: "Type text, then Enter" },
  hint: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const sortedValues = computed(() => parseStringList(props.modelValue));

const tagValues = computed(() => sortedValues.value);

// Only render the dropdown when there are real preset options to pick. Otherwise
// it is free-entry (user-typed tags only) and a caret would be misleading.
const hasPresetOptions = computed(
  () => (props.presetOptions || []).map((o) => String(o).trim()).filter(Boolean).length > 0
);

const presetOptions = computed(() => {
  const fromSchema = (props.presetOptions || []).map((o) => String(o).trim()).filter(Boolean);
  return [...new Set([...fromSchema, ...sortedValues.value])];
});

function onSelectChange(raw?: string[]): void {
  const strings = parseStringList(raw ?? []);
  emit("update:modelValue", strings.length ? strings : "");
}
</script>

<style scoped>
.field-full {
  width: 100%;
}
.list-hint {
  display: block;
  margin-top: 4px;
}
</style>

<template>
  <div class="string-list">
    <el-select
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
    <el-text v-if="hint" type="info" size="small" class="list-hint">{{ hint }}</el-text>
  </div>
</template>

<script setup>
import { computed } from "vue";
import { parseStringList } from "../lib/stringList";

const props = defineProps({
  modelValue: { type: [Array, String], default: () => [] },
  presetOptions: { type: Array, default: () => [] },
  placeholder: { type: String, default: "Type text, then Enter" },
  hint: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const sortedValues = computed(() => parseStringList(props.modelValue));

const tagValues = computed(() => sortedValues.value);

const presetOptions = computed(() => {
  const fromSchema = (props.presetOptions || []).map((o) => String(o).trim()).filter(Boolean);
  return [...new Set([...fromSchema, ...sortedValues.value])];
});

function onSelectChange(raw) {
  const strings = parseStringList(raw);
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

<template>
  <div class="tag-filter-builder">
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Has all</span>
      <el-select
        v-model="all"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        placeholder="every one of these tags"
        size="small"
        @change="emitFilter"
      >
        <el-option v-for="t in tagOptions" :key="t" :label="t" :value="t" />
      </el-select>
    </div>
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Has any</span>
      <el-select
        v-model="any"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        placeholder="at least one of these tags"
        size="small"
        @change="emitFilter"
      >
        <el-option v-for="t in tagOptions" :key="t" :label="t" :value="t" />
      </el-select>
    </div>
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Lacks</span>
      <el-select
        v-model="none"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        placeholder="none of these tags"
        size="small"
        @change="emitFilter"
      >
        <el-option v-for="t in tagOptions" :key="t" :label="t" :value="t" />
      </el-select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from "vue";
import type { TagEditOpDto } from "../types/api";

const props = defineProps<{
  modelValue: TagEditOpDto["filter"];
  tagOptions: string[];
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: TagEditOpDto["filter"]): void;
}>();

const all = ref<string[]>([...(props.modelValue?.all ?? [])]);
const any = ref<string[]>([...(props.modelValue?.any ?? [])]);
const none = ref<string[]>([...(props.modelValue?.none ?? [])]);

watch(
  () => props.modelValue,
  (value) => {
    all.value = [...(value?.all ?? [])];
    any.value = [...(value?.any ?? [])];
    none.value = [...(value?.none ?? [])];
  },
  { deep: true }
);

function emitFilter(): void {
  emit("update:modelValue", {
    all: [...all.value],
    any: [...any.value],
    none: [...none.value],
  });
}
</script>

<style scoped>
.tag-filter-builder {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.tag-filter-builder__row {
  display: grid;
  grid-template-columns: 64px 1fr;
  align-items: center;
  gap: 8px;
}
.tag-filter-builder__label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  text-align: right;
}
</style>

<template>
  <div class="tag-filter-builder">
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Has all</span>
      <el-select-v2
        v-model="all"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        :options="tagSelectOptions"
        placeholder="every one of these tags"
        size="small"
        @change="emitFilter"
      />
    </div>
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Has any</span>
      <el-select-v2
        v-model="any"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        :options="tagSelectOptions"
        placeholder="at least one of these tags"
        size="small"
        @change="emitFilter"
      />
    </div>
    <div class="tag-filter-builder__row">
      <span class="tag-filter-builder__label">Lacks</span>
      <el-select-v2
        v-model="none"
        multiple
        filterable
        allow-create
        default-first-option
        clearable
        :options="tagSelectOptions"
        placeholder="none of these tags"
        size="small"
        @change="emitFilter"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
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

// el-select-v2 virtualizes its option list (unlike el-select's el-option
// children), so large vocabularies don't block the main thread on mount.
const tagSelectOptions = computed(() => props.tagOptions.map((t) => ({ value: t, label: t })));

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

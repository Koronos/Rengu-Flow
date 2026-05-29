<template>
  <div class="kv-list">
    <div v-for="(row, index) in rows" :key="row._id" class="kv-row">
      <el-input
        v-model="row.key"
        placeholder="Parameter"
        class="kv-key"
        clearable
        @update:model-value="emitDict"
      />
      <el-autocomplete
        v-if="runtimeTokens.length"
        v-model="row.value"
        :fetch-suggestions="fetchValueSuggestions"
        clearable
        class="kv-value"
        placeholder="Value or runtime token"
        :trigger-on-focus="true"
        @update:model-value="emitDict"
        @select="emitDict"
      />
      <el-input
        v-else
        v-model="row.value"
        placeholder="Value"
        clearable
        class="kv-value"
        @update:model-value="emitDict"
      />
      <el-button
        type="danger"
        link
        :icon="Delete"
        :disabled="rows.length <= 1 && !row.key && !row.value"
        v-bind="ariaLabel('Remove parameter')"
        @click="removeRow(index)"
      />
    </div>
    <el-button type="primary" link class="kv-add" @click="addRow">Add parameter</el-button>
    <el-text v-if="hint" type="info" size="small" class="kv-hint">{{ hint }}</el-text>
  </div>
</template>

<script setup lang="ts">
import { Delete } from "@element-plus/icons-vue";
import { ref, watch } from "vue";
import {
  dictToKvRows,
  kvModelMatchesRows,
  kvRowsToDict,
  type KvRow,
} from "../lib/keyValueList";
import { ariaLabel } from "../lib/aria";
import type { PropType } from "vue";

type RowState = KvRow & { _id: number };

const props = defineProps({
  modelValue: { type: [Object, String] as PropType<unknown> },
  runtimeTokens: { type: Array as PropType<string[]>, default: () => [] },
  hint: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

let rowSeq = 0;
function newRow(key = "", value = ""): RowState {
  rowSeq += 1;
  return { _id: rowSeq, key, value };
}

const rows = ref<RowState[]>([newRow()]);

function syncRowsFromModel(): void {
  const parsed = dictToKvRows(props.modelValue);
  rows.value = parsed.length ? parsed.map((r) => newRow(r.key, r.value)) : [newRow()];
}

watch(
  () => props.modelValue,
  () => {
    if (kvModelMatchesRows(props.modelValue, rows.value)) return;
    syncRowsFromModel();
  },
  { immediate: true, deep: true }
);

function emitDict(): void {
  const dict = kvRowsToDict(rows.value);
  emit("update:modelValue", Object.keys(dict).length ? dict : "");
}

function addRow(): void {
  rows.value = [...rows.value, newRow()];
}

function removeRow(index: number): void {
  const next = rows.value.filter((_, i) => i !== index);
  rows.value = next.length ? next : [newRow()];
  emitDict();
}

function fetchValueSuggestions(
  query: string,
  cb: (items: { value: string }[]) => void
): void {
  const q = query.trim().toLowerCase();
  const tokens = props.runtimeTokens || [];
  const matches = tokens
    .filter((t) => !q || t.toLowerCase().includes(q))
    .map((value) => ({ value }));
  cb(matches);
}
</script>

<style scoped>
.kv-list {
  width: 100%;
}
.kv-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}
.kv-key {
  flex: 0 0 180px;
  max-width: 220px;
}
.kv-value {
  flex: 1;
  min-width: 0;
}
.kv-add {
  margin-top: 4px;
}
.kv-hint {
  display: block;
  margin-top: 6px;
}
</style>

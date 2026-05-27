<template>
  <div class="eval-datasets-field">
    <el-tag
      v-for="(entry, idx) in entries"
      :key="idx"
      closable
      class="eval-tag"
      @close="removeAt(idx)"
    >
      {{ entryLabel(entry) }}
    </el-tag>
    <el-space wrap>
      <el-button size="small" @click="pickerOpen = true">Add dataset…</el-button>
      <el-button size="small" link @click="showAdvanced = !showAdvanced">
        {{ showAdvanced ? "Hide" : "Edit as JSON" }}
      </el-button>
    </el-space>
    <el-input
      v-if="showAdvanced"
      :model-value="jsonText"
      type="textarea"
      :rows="4"
      class="field-full mt-8"
      @update:model-value="onJsonInput"
    />
    <DatasetPickerModal
      v-model="pickerOpen"
      multiple
      :selected="entryPaths"
      @select-multiple="onAddMultiple"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, type PropType } from "vue";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import { canonicalDatasetRef, datasetRefDisplayLabel } from "../lib/datasetLibraryRef";
import { jsonStringify } from "../lib/formUtils";

export type EvalDatasetEntry = string | { config?: string; name?: string; [key: string]: unknown };

const props = defineProps({
  modelValue: {
    type: [Array, String, Object] as PropType<EvalDatasetEntry[] | EvalDatasetEntry | null>,
    default: () => [],
  },
});

const emit = defineEmits(["update:modelValue"]);

const pickerOpen = ref(false);
const showAdvanced = ref(false);

const entries = computed(() => {
  const v = props.modelValue;
  if (Array.isArray(v)) return v;
  if (v === undefined || v === null || v === "") return [];
  return [v];
});

const entryPaths = computed(() =>
  entries.value.map((e) => (typeof e === "string" ? e : e?.config || e?.name || "")).filter(Boolean)
);

const jsonText = computed(() => jsonStringify(entries.value));

function entryLabel(entry: EvalDatasetEntry) {
  if (typeof entry === "string") {
    return datasetRefDisplayLabel(entry);
  }
  return entry?.name || entry?.config || JSON.stringify(entry);
}

function emitEntries(next: EvalDatasetEntry[]) {
  emit("update:modelValue", next);
}

function removeAt(idx) {
  const next = entries.value.filter((_, i) => i !== idx);
  emitEntries(next);
}

function onAddMultiple(paths: string[]) {
  const existing = new Set(entryPaths.value.map(canonicalDatasetRef));
  const next = [...entries.value];
  for (const p of paths) {
    const key = canonicalDatasetRef(p);
    if (!existing.has(key)) {
      next.push(p);
      existing.add(key);
    }
  }
  emitEntries(next);
}

function onJsonInput(text) {
  try {
    const parsed = JSON.parse(text || "[]");
    emitEntries(Array.isArray(parsed) ? parsed : []);
  } catch {
    /* keep typing */
  }
}
</script>

<style scoped>
.eval-tag {
  margin: 0 6px 6px 0;
}
.mt-8 {
  margin-top: 8px;
}
</style>

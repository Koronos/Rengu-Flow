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
    <el-empty v-if="!entries.length" description="No eval datasets yet" :image-size="56">
      <el-button type="primary" :icon="Plus" @click="pickerOpen = true">Add eval dataset</el-button>
    </el-empty>
    <el-space v-else wrap>
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
      :placeholder="jsonPlaceholder"
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
import { Plus } from "@element-plus/icons-vue";
import DatasetPickerModal from "./DatasetPickerModal.vue";
import { appendUniqueDatasetPaths } from "../lib/datasetLibraryRef";
import { useResolvedDatasetLabels } from "../composables/useResolvedDatasetLabels";
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

// Shown in the raw-JSON editor so users know the expected shape: an array whose
// entries are either a dataset path string or a { config, name } object.
const jsonPlaceholder =
  '[\n  "datasets/eval.toml",\n  { "config": "datasets/eval-hard.toml", "name": "Hard set" }\n]';

const entries = computed(() => {
  const v = props.modelValue;
  if (Array.isArray(v)) return v;
  if (v === undefined || v === null || v === "") return [];
  return [v];
});

const entryPaths = computed(() =>
  entries.value.map((e) => (typeof e === "string" ? e : e?.config || e?.name || "")).filter(Boolean)
);

const { labelFor } = useResolvedDatasetLabels(entryPaths);

const jsonText = computed(() => jsonStringify(entries.value));

function entryLabel(entry: EvalDatasetEntry) {
  if (typeof entry === "string") {
    return labelFor(entry);
  }
  return entry?.name || (entry?.config ? labelFor(String(entry.config)) : "") || JSON.stringify(entry);
}

function emitEntries(next: EvalDatasetEntry[]) {
  emit("update:modelValue", next);
}

function removeAt(idx: number) {
  const next = entries.value.filter((_, i) => i !== idx);
  emitEntries(next);
}

function onAddMultiple(paths: string[]) {
  const objects = entries.value.filter((e) => typeof e !== "string");
  const strings = entries.value.filter((e): e is string => typeof e === "string");
  emitEntries([...objects, ...appendUniqueDatasetPaths(strings, paths)]);
}

function onJsonInput(text: string) {
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

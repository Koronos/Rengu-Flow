<template>
  <div class="mb-field">
    <el-radio-group :model-value="model.mode" size="small" @update:model-value="onMode">
      <el-radio-button value="uniform">Uniform</el-radio-button>
      <el-radio-button value="per_resolution">Per resolution</el-radio-button>
    </el-radio-group>

    <el-input-number
      v-if="model.mode === 'uniform'"
      :model-value="model.uniform"
      :min="1"
      :step="1"
      controls-position="right"
      class="mb-uniform"
      @update:model-value="onUniform"
    />

    <template v-else>
      <div v-for="(row, index) in model.rows" :key="index" class="mb-row">
        <el-input-number
          :model-value="row.resolution"
          :min="1"
          :step="64"
          controls-position="right"
          placeholder="Resolution"
          class="mb-res"
          @update:model-value="(v: number | undefined) => onRow(index, 'resolution', v)"
        />
        <el-input-number
          :model-value="row.batch"
          :min="1"
          :step="1"
          controls-position="right"
          placeholder="Batch"
          class="mb-bs"
          @update:model-value="(v: number | undefined) => onRow(index, 'batch', v)"
        />
        <el-button
          type="danger"
          link
          :icon="Delete"
          :disabled="model.rows.length <= 1"
          v-bind="ariaLabel('Remove resolution row')"
          @click="removeRow(index)"
        />
      </div>
      <el-button type="primary" link class="mb-add" @click="addRow">Add resolution</el-button>
      <el-text v-for="issue in issues" :key="issue" type="warning" size="small" class="mb-issue">
        {{ issue }}
      </el-text>
      <el-text type="info" size="small" class="mb-hint">
        Buckets use the numerically closest resolution; other resolutions fall back to it.
      </el-text>
    </template>
  </div>
</template>

<script setup lang="ts">
import { Delete } from "@element-plus/icons-vue";
import { computed, ref, watch } from "vue";
import {
  microBatchIssues,
  parseMicroBatch,
  serializeMicroBatch,
  type MicroBatchMode,
  type MicroBatchModel,
} from "../lib/microBatchMap";
import { ariaLabel } from "../lib/aria";
import type { PropType } from "vue";

const props = defineProps({
  modelValue: { type: [Number, Object, String] as PropType<unknown> },
});

const emit = defineEmits(["update:modelValue"]);

const model = ref<MicroBatchModel>(parseMicroBatch(props.modelValue));

watch(
  () => props.modelValue,
  (value) => {
    // Ignore the echo of our own emit (serialized forms match).
    if (JSON.stringify(serializeMicroBatch(model.value) ?? null) === JSON.stringify(value ?? null)) {
      return;
    }
    model.value = parseMicroBatch(value);
  },
  { deep: true },
);

const issues = computed(() => microBatchIssues(model.value));

function emitValue(): void {
  emit("update:modelValue", serializeMicroBatch(model.value));
}

function onMode(value: string | number | boolean | undefined): void {
  const mode = value as MicroBatchMode;
  if (mode !== "uniform" && mode !== "per_resolution") return;
  if (mode === model.value.mode) return;
  if (mode === "per_resolution") {
    // Seed the table from the current uniform value so switching is lossless.
    model.value = {
      mode,
      uniform: undefined,
      rows: [{ resolution: undefined, batch: model.value.uniform ?? 1 }],
    };
  } else {
    const first = model.value.rows.find((r) => r.batch !== undefined);
    model.value = { mode, uniform: first?.batch ?? 1, rows: [] };
  }
  emitValue();
}

function onUniform(v: number | undefined): void {
  model.value = { ...model.value, uniform: v ?? undefined };
  emitValue();
}

function onRow(index: number, key: "resolution" | "batch", v: number | undefined): void {
  const rows = model.value.rows.slice();
  rows[index] = { ...rows[index], [key]: v ?? undefined };
  model.value = { ...model.value, rows };
  emitValue();
}

function addRow(): void {
  model.value = {
    ...model.value,
    rows: [...model.value.rows, { resolution: undefined, batch: 1 }],
  };
}

function removeRow(index: number): void {
  const rows = model.value.rows.filter((_, i) => i !== index);
  model.value = { ...model.value, rows: rows.length ? rows : [{ resolution: undefined, batch: 1 }] };
  emitValue();
}
</script>

<style scoped>
.mb-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
}
.mb-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.mb-res,
.mb-bs {
  width: 140px;
}
.mb-uniform {
  width: 140px;
}
</style>

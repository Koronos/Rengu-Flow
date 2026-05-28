<template>
  <div class="strategy-override-editor">
    <el-form-item>
      <template #label>
        <span class="label-row">
          <span>Enabled</span>
          <FieldHelpIcon v-if="enabledHelpField" :field="enabledHelpField" />
        </span>
      </template>
      <el-switch
        :model-value="params.enabled !== false"
        @update:model-value="(v) => emitParam('enabled', Boolean(v))"
      />
    </el-form-item>

    <el-row v-if="strategy?.parameters?.length" :gutter="16">
      <el-col
        v-for="field in strategy.parameters"
        :key="field.path"
        :xs="24"
        :sm="12"
      >
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>{{ field.label }}</span>
              <FieldHelpIcon :field="paramField(field)" />
            </span>
          </template>
          <el-select
            v-if="field.type === 'select'"
            :model-value="String(params[field.path] ?? field.default ?? '')"
            class="field-full"
            @update:model-value="(v) => emitParam(field.path, v)"
          >
            <el-option
              v-for="opt in field.options || []"
              :key="opt"
              :label="opt"
              :value="opt"
            />
          </el-select>
          <el-input-number
            v-else-if="field.type === 'integer'"
            :model-value="num(params[field.path], field.default)"
            :min="field.min"
            :max="field.max"
            :step="field.step ?? 1"
            class="field-narrow"
            @update:model-value="(v) => emitParam(field.path, v == null ? undefined : Math.round(Number(v)))"
          />
          <el-input-number
            v-else
            :model-value="num(params[field.path], field.default)"
            :min="field.min"
            :max="field.max"
            :step="field.step ?? 0.01"
            class="field-narrow"
            @update:model-value="(v) => emitParam(field.path, v == null ? undefined : Number(v))"
          />
        </el-form-item>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import type { AugParamField, AugStrategyCatalogEntry } from "../lib/datasetAugmentation";
import type { SchemaField } from "../types/forms";

const props = defineProps<{
  strategy?: AugStrategyCatalogEntry;
  params: Record<string, unknown>;
}>();

const emit = defineEmits<{
  (e: "update", value: Record<string, unknown>): void;
}>();

const enabledHelpField = computed<SchemaField | null>(() => {
  const help = props.strategy?.help?.trim();
  if (!help) return null;
  return {
    path: `${props.strategy?.name || "strategy"}.enabled`,
    label: "Enabled",
    type: "boolean",
    help,
    doc_path: "docs/user/dataset-augmentation.md",
  };
});

function paramField(field: AugParamField): SchemaField {
  return {
    path: field.path,
    label: field.label,
    type: field.type === "select" ? "select" : field.type === "integer" ? "integer" : "number",
    help: field.help || field.label,
    doc_path: field.help ? "docs/user/dataset-augmentation.md" : undefined,
  };
}

function num(value: unknown, fallback?: number | string | boolean): number {
  if (value === "" || value == null) {
    const f = Number(fallback);
    return Number.isFinite(f) ? f : 0;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function emitParam(path: string, value: unknown) {
  emit("update", { ...props.params, [path]: value });
}
</script>

<style scoped>
.strategy-override-editor :deep(.el-form-item) {
  margin-bottom: 10px;
}
.label-row {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
</style>

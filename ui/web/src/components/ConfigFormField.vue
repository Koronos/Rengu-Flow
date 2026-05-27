<template>
  <el-form-item
    v-if="visible"
    :class="{
      'field-required': field.required,
      'field-recommended': field.importance === 'recommended' || field.recommended,
    }"
  >
    <template #label>
      <span class="label-row">
        <span>{{ field.label }}</span>
        <el-tag v-if="field.required" type="danger" size="small" effect="plain" class="req-tag">
          required
        </el-tag>
        <el-tag
          v-else-if="field.importance === 'recommended' || field.recommended"
          type="warning"
          size="small"
          effect="plain"
          class="req-tag"
        >
          important
        </el-tag>
        <el-text
          v-if="hasDefault"
          type="info"
          size="small"
          class="default-hint"
        >
          default: {{ formatDefault(field.default) }}
        </el-text>
        <FieldHelpIcon :field="field" />
      </span>
      <el-text type="info" size="small" class="path-tag">{{ field.path }}</el-text>
    </template>

    <TrainingDatasetsField
      v-if="field.path === 'dataset'"
      :model-value="trainingDatasetModel"
      @update:model-value="onTrainingDatasetsInput"
    />

    <EvalDatasetsField
      v-else-if="field.path === 'eval_datasets'"
      :model-value="evalDatasetsModel"
      @update:model-value="onEvalDatasetsInput"
    />

    <el-autocomplete
      v-else-if="field.type === 'select' && field.allow_custom"
      v-model="editableText"
      :fetch-suggestions="fetchSuggestions"
      clearable
      :class="widthClass"
      placeholder="Edit in place or pick a suggestion…"
      :trigger-on-focus="true"
      :select-when-unmatched="false"
      highlight-first-item
      @clear="onClear"
      @select="onAutocompleteSelect"
    />

    <el-select
      v-else-if="field.type === 'select'"
      :model-value="stringValue"
      clearable
      filterable
      :class="widthClass"
      @update:model-value="onInput"
    >
      <el-option
        v-for="opt in selectOptions"
        :key="opt.value"
        :label="opt.label"
        :value="opt.value"
      />
    </el-select>

    <el-switch
      v-else-if="field.type === 'boolean'"
      :model-value="!!effectiveValue"
      @update:model-value="onBooleanInput"
    />

    <el-input-number
      v-else-if="field.type === 'integer'"
      :model-value="numberValue"
      :min="field.min ?? undefined"
      :step="1"
      controls-position="right"
      class="field-narrow"
      @update:model-value="onInput"
    />

    <el-input-number
      v-else-if="field.type === 'number'"
      :model-value="numberValue"
      :step="0.0001"
      controls-position="right"
      class="field-narrow"
      @update:model-value="onInput"
    />

    <IntegerListField
      v-else-if="field.type === 'integer_list'"
      :model-value="listModelValue"
      :preset-options="field.options || []"
      :placeholder="field.placeholder || 'Pick or type a number, then Enter'"
      :min="field.min ?? 1"
      @update:model-value="onIntegerListInput"
    />

    <NumberListField
      v-else-if="field.type === 'number_list'"
      :model-value="listModelValue"
      :preset-options="field.options || []"
      :placeholder="field.placeholder || 'Type a number, then Enter'"
      :min="field.min"
      :max="field.max"
      @update:model-value="onNumberListInput"
    />

    <StringListField
      v-else-if="field.type === 'string_list' && !stringListUseJson"
      :model-value="stringListModel"
      :preset-options="field.options || []"
      :placeholder="field.placeholder || 'Type text, then Enter'"
      :hint="field.string_list_hint || ''"
      @update:model-value="onStringListInput"
    />

    <el-input
      v-else-if="field.type === 'json' || stringListUseJson"
      :model-value="displayValue"
      type="textarea"
      :rows="4"
      :class="widthClass"
      @update:model-value="onInput"
    />

    <el-input
      v-else
      :model-value="displayValue"
      :placeholder="field.placeholder"
      clearable
      :class="widthClass"
      @update:model-value="onInput"
    />

    <div v-if="resolveHint" class="resolve-hint">
      <el-text v-if="resolveHint.loading" type="info" size="small">Checking availability…</el-text>
      <el-text v-else-if="resolveHint.available" type="success" size="small">
        {{ resolveHint.text }}
      </el-text>
      <el-text v-else type="danger" size="small">{{ resolveHint.text }}</el-text>
    </div>
  </el-form-item>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  adapterOptionsForModel,
  datasetFieldVisible,
  fieldEffectiveValue,
  fieldVisible,
  jsonStringify,
} from "../lib/formUtils";
import { api } from "../api";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import TrainingDatasetsField from "./TrainingDatasetsField.vue";
import EvalDatasetsField, { type EvalDatasetEntry } from "./EvalDatasetsField.vue";
import IntegerListField from "./IntegerListField.vue";
import NumberListField from "./NumberListField.vue";
import StringListField from "./StringListField.vue";
import { integerListToFormValue } from "../lib/integerList";
import { numberListToFormValue } from "../lib/numberList";
import { stringListNeedsJsonEditor, stringListToFormValue } from "../lib/stringList";
import type { PropType } from "vue";
import type { FormValues, ModelCapabilities, RawListInput, SchemaField } from "../types/forms";

const props = defineProps({
  field: { type: Object as PropType<SchemaField>, required: true },
  form: { type: Object as PropType<FormValues>, required: true },
  capabilities: { type: Object as PropType<ModelCapabilities>, default: () => ({}) },
  /** When true, always render (parent already decided visibility). */
  alwaysVisible: { type: Boolean, default: false },
  /** Dataset TOML form (flat keys, no training config visibility). */
  datasetForm: { type: Boolean, default: false },
});

const emit = defineEmits(["update:path"]);

const visible = computed(() => {
  if (props.alwaysVisible) return true;
  const f = props.field;
  if (props.datasetForm || !f.path?.includes(".")) {
    return datasetFieldVisible(f, props.form);
  }
  return fieldVisible(f, props.form, props.capabilities);
});

const stringListUseJson = computed(() => {
  if (props.field.type !== "string_list") return false;
  const raw = (props.field.path in props.form
    ? props.form[props.field.path]
    : effectiveValue.value) as RawListInput;
  return stringListNeedsJsonEditor(raw);
});

const hasDefault = computed(
  () =>
    props.field.default !== undefined &&
    props.field.default !== null &&
    props.field.default !== ""
);

const effectiveValue = computed(() => fieldEffectiveValue(props.field, props.form));

const listModelValue = computed(
  () => effectiveValue.value as string | number | unknown[] | undefined
);

const stringListModel = computed(
  () => effectiveValue.value as string | unknown[] | undefined
);

const evalDatasetsModel = computed(
  () => effectiveValue.value as EvalDatasetEntry[] | EvalDatasetEntry | null | undefined
);

const trainingDatasetModel = computed(() => {
  const v = effectiveValue.value;
  return v === undefined || v === null ? "" : String(v);
});

const selectOptions = computed(() => {
  if (props.field.options_from_model) {
    return adapterOptionsForModel(props.capabilities, props.form["model.type"]).map((o) => ({
      label: String(o),
      value: String(o),
    }));
  }
  const values = props.field.option_values || [];
  const labels = props.field.options || [];
  if (values?.length) {
    return values.map((v, i) => ({
      value: String(v),
      label: String(labels[i] ?? v),
    }));
  }
  return labels.map((o) => ({ label: String(o), value: String(o) }));
});

const stringValue = computed(() => {
  const v = effectiveValue.value;
  return v === undefined || v === null ? "" : String(v);
});

/** v-model for autocomplete so clearable (×) syncs correctly. */
const editableText = computed({
  get: () => stringValue.value,
  set: (val) => onInput(val ?? ""),
});

const numberValue = computed(() => {
  const v = effectiveValue.value;
  if (v === "" || v === undefined || v === null) return undefined;
  const n = Number(v);
  return Number.isNaN(n) ? undefined : n;
});

const displayValue = computed(() => {
  const v = effectiveValue.value;
  if (props.field.type === "json") return jsonStringify(v);
  if (v === undefined || v === null) return "";
  return String(v);
});

function isPathField(field) {
  const p = field.path || "";
  if (p === "dataset" || p === "eval_datasets") return false;
  if (p.includes("path") || p.endsWith("_dir") || p === "output_dir" || p === "resume_from") {
    return true;
  }
  return false;
}

const widthClass = computed(() => {
  const f = props.field;
  if (f.path === "dataset" || f.path === "eval_datasets") return "";
  if (f.type === "integer" || f.type === "number") return "field-narrow";
  if (isPathField(f)) return "field-path";
  if (f.type === "boolean") return "";
  return "field-full";
});

const resolveHint = ref(null);
let probeTimer = null;

function formatDefault(val) {
  if (typeof val === "boolean") return val ? "true" : "false";
  return String(val);
}

function onInput(val) {
  const next = val === undefined || val === null ? "" : val;
  emit("update:path", { path: props.field.path, value: next });
}

function onBooleanInput(val) {
  emit("update:path", { path: props.field.path, value: !!val });
}

function onIntegerListInput(val) {
  emit("update:path", {
    path: props.field.path,
    value: integerListToFormValue(val),
  });
}

function onNumberListInput(val) {
  emit("update:path", {
    path: props.field.path,
    value: numberListToFormValue(val),
  });
}

function onStringListInput(val) {
  emit("update:path", {
    path: props.field.path,
    value: stringListToFormValue(val),
  });
}

function onEvalDatasetsInput(val) {
  emit("update:path", { path: props.field.path, value: val });
}

function onTrainingDatasetsInput(val) {
  emit("update:path", { path: props.field.path, value: val });
}

function onClear() {
  onInput("");
  resolveHint.value = null;
}

function fetchSuggestions(queryString, cb) {
  const opts = selectOptions.value.map((o) => o.value);
  const q = (queryString || "").trim().toLowerCase();
  const matches = q ? opts.filter((o) => o.toLowerCase().includes(q)) : opts;
  cb(matches.slice(0, 80).map((value) => ({ value })));
}

function onAutocompleteSelect(item) {
  if (item?.value !== undefined) {
    onInput(item.value);
  }
}

async function runProbe(name) {
  if (!props.field.allow_custom || !name?.trim()) {
    resolveHint.value = null;
    return;
  }
  resolveHint.value = { loading: true, text: "" };
  try {
    const body =
      props.field.path === "optimizer.type"
        ? { optimizer: name.trim() }
        : props.field.path === "lr_scheduler"
          ? { scheduler: name.trim() }
          : null;
    if (!body) {
      resolveHint.value = null;
      return;
    }
    const r = (await api.probeRegistry(body)) as {
      optimizer?: { available?: boolean; resolved_class?: string; resolved?: string; source?: string; error?: string };
      scheduler?: { available?: boolean; resolved_class?: string; resolved?: string; source?: string; error?: string };
    };
    const check = r.optimizer || r.scheduler;
    if (check?.available) {
      const detail = check.resolved_class || check.resolved || check.source;
      resolveHint.value = {
        loading: false,
        available: true,
        text: detail ? `Available → ${detail}` : "Available in this environment",
      };
    } else {
      resolveHint.value = {
        loading: false,
        available: false,
        text: check?.error || "Not available",
      };
    }
  } catch (e) {
    resolveHint.value = {
      loading: false,
      available: false,
      text: String(e),
    };
  }
}

watch(stringValue, (v) => {
  if (!props.field.allow_custom) return;
  clearTimeout(probeTimer);
  probeTimer = setTimeout(() => runProbe(v), 450);
});
</script>

<style scoped>
.label-row {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
.req-tag {
  margin-left: 2px;
}
.path-tag {
  display: block;
  font-family: ui-monospace, monospace;
  margin-top: 2px;
}
.field-required :deep(.el-form-item__label),
.field-recommended :deep(.el-form-item__label) {
  font-weight: 600;
}
.default-hint {
  font-family: ui-monospace, monospace;
  font-size: 11px;
}
.resolve-hint {
  margin-top: 4px;
  line-height: 1.4;
  overflow-wrap: anywhere;
  word-break: break-word;
}
</style>

<template>
  <el-form-item
    v-if="visible"
    :class="{
      'field-required': field.required,
      'field-optional': isOptionalField,
    }"
  >
    <template v-if="!hideLabel" #label>
      <span class="label-row">
        <span>{{ field.label }}</span>
        <span
          v-if="field.required"
          class="rf-label-required"
          aria-hidden="true"
          title="Required"
        >*</span>
        <span v-if="isOptionalField" class="rf-label-optional-hint">(optional)</span>
        <el-text
          v-if="hasDefault"
          type="info"
          size="small"
          class="default-hint"
        >
          default: {{ formatDefault(field.default) }}
        </el-text>
        <FieldHelpIcon v-if="!hideLabelHelp" :field="field" />
      </span>
      <FieldPathTag v-if="pathTagPlacement === 'label'" :path="fieldPathTag" />
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
      @update:model-value="(val) => onBooleanInput(Boolean(val))"
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
      :min="numberMin"
      :max="numberMax"
      :step="numberStep"
      :value-on-clear="null"
      controls-position="right"
      class="field-narrow"
      @update:model-value="onInput"
    />

    <NumericListField
      v-else-if="field.type === 'integer_list'"
      integer
      :model-value="listModelValue"
      :preset-options="(field.options || []) as Array<string | number>"
      :placeholder="field.placeholder || 'Pick or type a number, then Enter'"
      :min="field.min ?? 1"
      @update:model-value="onListInput"
    />

    <NumericListField
      v-else-if="field.type === 'number_list'"
      :model-value="listModelValue"
      :preset-options="(field.options || []) as Array<string | number>"
      :placeholder="field.placeholder || 'Type a number, then Enter'"
      :min="field.min"
      :max="field.max"
      :max-length="field.max_length"
      @update:model-value="onListInput"
    />

    <StringListField
      v-else-if="field.type === 'string_list' && !stringListUseJson"
      :model-value="stringListModel"
      :preset-options="(field.options || []) as Array<string | number>"
      :placeholder="field.placeholder || 'Type text, then Enter'"
      :hint="field.string_list_hint || ''"
      @update:model-value="onListInput"
    />

    <SizeBucketsField
      v-else-if="field.path === 'size_buckets'"
      :model-value="sizeBucketsModel"
      @update:model-value="onSizeBucketsInput"
    />

    <KeyValueListField
      v-else-if="field.type === 'key_value_list'"
      :model-value="effectiveValue"
      :runtime-tokens="field.runtime_tokens || []"
      :hint="field.string_list_hint || ''"
      @update:model-value="onKvInput"
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

    <div v-if="pathLoading || pathError || pathOk" class="resolve-hint">
      <PathValidationFeedback
        :loading="pathLoading"
        :error="pathError"
        :ok="pathOk"
      />
    </div>
    <div v-else-if="resolveHint" class="resolve-hint">
      <el-text v-if="resolveHint.loading" type="info" size="small">Checking availability…</el-text>
      <el-text v-else-if="resolveHint.available" type="success" size="small">
        {{ resolveHint.text }}
      </el-text>
      <el-text v-else type="danger" size="small">{{ resolveHint.text }}</el-text>
    </div>

    <FieldPathTag
      v-if="pathTagPlacement === 'foot'"
      :path="fieldPathTag"
      class="path-tag--foot"
    />
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
import { isPathField, pathFieldExpect } from "../lib/pathFields";
import { usePathValidation } from "../composables/usePathValidation";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import FieldPathTag from "./FieldPathTag.vue";
import PathValidationFeedback from "./PathValidationFeedback.vue";
import TrainingDatasetsField from "./TrainingDatasetsField.vue";
import {
  coerceTrainingDatasetEntries,
  trainingDatasetFormValue,
} from "../lib/datasetLibraryRef";
import EvalDatasetsField, { type EvalDatasetEntry } from "./EvalDatasetsField.vue";
import NumericListField from "./NumericListField.vue";
import SizeBucketsField from "./SizeBucketsField.vue";
import StringListField from "./StringListField.vue";
import KeyValueListField from "./KeyValueListField.vue";
import type { SizeBucket } from "../lib/sizeBuckets";
import { listToFormValue } from "../lib/listToFormValue";
import { formatDefaultValue } from "../lib/defaultFormat";
import { stringListNeedsJsonEditor } from "../lib/stringList";
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
  /** Dataset root values for directory-row `show_when_field` parents. */
  directoryInheritForm: { type: Object as PropType<FormValues>, default: null },
  /** Parent row already shows the field label (directory override toggle). */
  hideLabel: { type: Boolean, default: false },
  /** Suppress help icon in the label slot when the parent shows it. */
  hideLabelHelp: { type: Boolean, default: false },
  /** Where to render the TOML path hint (`label` = under title, `foot` = below control). */
  pathTagPlacement: {
    type: String as PropType<"label" | "foot" | "none">,
    default: "label",
  },
});

const emit = defineEmits(["update:path"]);

const visible = computed(() => {
  if (props.alwaysVisible) return true;
  const f = props.field;
  if (props.datasetForm) {
    const inherit =
      props.directoryInheritForm != null ? props.directoryInheritForm : undefined;
    return datasetFieldVisible(f, props.form, inherit ? { inheritForm: inherit } : undefined);
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

const isOptionalField = computed(() => {
  const f = props.field;
  if (f.required || f.importance === "required") return false;
  if (f.importance === "recommended" || f.recommended) return false;
  // A field with a default shows `default: <value>` instead of "(optional)";
  // the two indicators are mutually exclusive.
  if (hasDefault.value) return false;
  return f.importance === "advanced";
});

const effectiveValue = computed(() => fieldEffectiveValue(props.field, props.form));

const listModelValue = computed(
  () => effectiveValue.value as string | number | unknown[] | undefined
);

const stringListModel = computed(
  () => effectiveValue.value as string | unknown[] | undefined
);

const sizeBucketsModel = computed(
  () => effectiveValue.value as SizeBucket[] | string | null | undefined
);

const evalDatasetsModel = computed(
  () => effectiveValue.value as EvalDatasetEntry[] | EvalDatasetEntry | null | undefined
);

const trainingDatasetModel = computed(() =>
  trainingDatasetFormValue(coerceTrainingDatasetEntries(effectiveValue.value))
);

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

const isSubsampleRatio = computed(() => props.field.path === "subsample_ratio");

/** Shorter path hint for KV fields (TOML target differs from form path). */
const fieldPathTag = computed(() => {
  const path = props.field.path;
  if (path === "lr_scheduler_args.extra_params") {
    return "lr_scheduler_args";
  }
  if (path === "optimizer.extra_params") {
    return "merged into [optimizer] in TOML";
  }
  return path;
});

const numberMin = computed(() => {
  if (isSubsampleRatio.value) return 0.0001;
  return props.field.min ?? undefined;
});

const numberMax = computed(() => {
  if (isSubsampleRatio.value) return 1;
  return props.field.max ?? undefined;
});

const numberStep = computed(() => (isSubsampleRatio.value ? 0.05 : 0.0001));

const displayValue = computed(() => {
  const v = effectiveValue.value;
  if (props.field.type === "json") return jsonStringify(v);
  if (v === undefined || v === null) return "";
  return String(v);
});

interface ResolveHint {
  loading: boolean;
  available?: boolean;
  text: string;
}

function isRegistryProbeField(field: SchemaField): boolean {
  return !!field.allow_custom && field.path === "lr_scheduler";
}

const pathValidationEnabled = computed(
  () => visible.value && isPathField(props.field) && !isRegistryProbeField(props.field)
);

const {
  loading: pathLoading,
  error: pathError,
  ok: pathOk,
  scheduleValidation: schedulePathValidation,
  clear: clearPathValidation,
} = usePathValidation({
  expect: () => (pathValidationEnabled.value ? pathFieldExpect(props.field) : null),
  required: () => !!(props.field.required && pathValidationEnabled.value),
});

const widthClass = computed(() => {
  const f = props.field;
  if (f.path === "dataset" || f.path === "eval_datasets") return "";
  if (f.type === "integer" || f.type === "number") return "field-narrow";
  if (isPathField(f)) return "field-path";
  if (f.type === "boolean") return "";
  return "field-full";
});

const resolveHint = ref<ResolveHint | null>(null);
let probeTimer: ReturnType<typeof setTimeout> | null = null;

function formatDefault(val: unknown): string {
  return formatDefaultValue(val);
}

function onInput(val: string | number | undefined | null): void {
  const next = val === undefined || val === null ? "" : val;
  emit("update:path", { path: props.field.path, value: next });
}

function onKvInput(val: unknown): void {
  emit("update:path", { path: props.field.path, value: val });
}

function onBooleanInput(val: boolean): void {
  emit("update:path", { path: props.field.path, value: !!val });
}

function onListInput(val: unknown[]): void {
  emit("update:path", {
    path: props.field.path,
    value: listToFormValue(val),
  });
}

function onSizeBucketsInput(val: SizeBucket[] | string): void {
  emit("update:path", { path: props.field.path, value: val });
}

function onEvalDatasetsInput(val: EvalDatasetEntry[] | EvalDatasetEntry | null): void {
  emit("update:path", { path: props.field.path, value: val });
}

function onTrainingDatasetsInput(val: string | string[]): void {
  emit("update:path", {
    path: props.field.path,
    value: trainingDatasetFormValue(coerceTrainingDatasetEntries(val)),
  });
}

function onClear() {
  onInput("");
  resolveHint.value = null;
  clearPathValidation();
}

function fetchSuggestions(
  queryString: string,
  cb: (items: { value: string }[]) => void
): void {
  const opts = selectOptions.value.map((o) => o.value);
  const q = (queryString || "").trim().toLowerCase();
  const matches = q ? opts.filter((o) => o.toLowerCase().includes(q)) : opts;
  cb(matches.slice(0, 80).map((value) => ({ value })));
}

function onAutocompleteSelect(item: { value?: string }): void {
  if (item?.value !== undefined) {
    onInput(item.value);
  }
}

async function runProbe(name: string): Promise<void> {
  if (!props.field.allow_custom || !name?.trim()) {
    resolveHint.value = null;
    return;
  }
  resolveHint.value = { loading: true, text: "" };
  try {
    const trimmed = name.trim();
    const body = props.field.path === "lr_scheduler" ? { scheduler: trimmed } : null;
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
      text: formatError(e),
    };
  }
}

watch(stringValue, (v) => {
  if (pathValidationEnabled.value) {
    schedulePathValidation(v);
    return;
  }
  if (!isRegistryProbeField(props.field)) return;
  if (probeTimer) clearTimeout(probeTimer);
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

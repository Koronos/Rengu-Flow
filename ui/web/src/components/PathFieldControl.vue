<template>
  <div class="path-field-control">
    <el-input
      :model-value="modelValue"
      :placeholder="placeholder"
      :clearable="clearable"
      :disabled="disabled"
      :class="inputClass"
      v-bind="inputAttrs"
      @update:model-value="onInput"
      @blur="onBlur"
      @keydown.enter="emit('enter')"
      @change="emit('change', $event)"
    >
      <template v-if="$slots.append" #append>
        <slot name="append" />
      </template>
    </el-input>
    <PathValidationFeedback
      :loading="loading"
      :error="error"
      :ok="showOk && ok"
      :ok-text="okText"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, useAttrs, watch } from "vue";
import PathValidationFeedback from "./PathValidationFeedback.vue";
import { usePathValidation } from "../composables/usePathValidation";
import type { PathExpect } from "../lib/pathFields";

defineOptions({ inheritAttrs: false });

const props = withDefaults(
  defineProps<{
    modelValue?: string;
    placeholder?: string;
    clearable?: boolean;
    disabled?: boolean;
    inputClass?: string;
    expect?: PathExpect | null;
    required?: boolean;
    validateOnBlur?: boolean;
    showOk?: boolean;
  }>(),
  {
    modelValue: "",
    clearable: true,
    disabled: false,
    inputClass: "field-path",
    expect: null,
    required: false,
    validateOnBlur: true,
    showOk: false,
  }
);

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
  (e: "blur"): void;
  (e: "enter"): void;
  (e: "change", value: string): void;
}>();

const attrs = useAttrs();
const inputAttrs = computed(() => {
  const { class: _class, ...rest } = attrs;
  return rest;
});

const { loading, error, ok, scheduleValidation, validate, clear } = usePathValidation({
  expect: () => props.expect,
  required: () => props.required,
});

const okText = computed(() => (props.showOk ? "Path found" : ""));

watch(
  () => props.modelValue,
  (value) => {
    scheduleValidation(value ?? "");
  }
);

function onInput(value: string | undefined | null): void {
  emit("update:modelValue", value ?? "");
}

function onBlur(): void {
  emit("blur");
  if (props.validateOnBlur) {
    void validate(props.modelValue ?? "");
  }
}

watch(
  () => props.modelValue,
  (value) => {
    if (!(value || "").trim()) clear();
  }
);
</script>

<style scoped>
.path-field-control {
  width: 100%;
}
</style>

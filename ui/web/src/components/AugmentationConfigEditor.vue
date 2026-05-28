<template>
  <div class="augmentation-config-editor">
    <el-row v-if="!hideEnable || !hidePreset" :gutter="16">
      <el-col v-if="!hideEnable" :xs="24" :sm="8">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>Enable augmentation</span>
              <FieldHelpIcon :field="fieldFor('enabled')" />
            </span>
          </template>
          <el-switch
            :model-value="Boolean(local.enabled)"
            @update:model-value="(v) => patch({ enabled: Boolean(v) })"
          />
        </el-form-item>
      </el-col>
      <el-col v-if="!hidePreset" :xs="24" :sm="hideEnable ? 24 : 16">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>Preset</span>
              <FieldHelpIcon :field="fieldFor('preset')" />
            </span>
          </template>
          <el-select
            :model-value="local.preset || 'none'"
            class="field-full"
            :disabled="!local.enabled"
            @update:model-value="(v) => patch({ preset: String(v) })"
          >
            <el-option
              v-for="preset in presetOptions"
              :key="preset.name"
              :label="preset.label"
              :value="preset.name"
            />
          </el-select>
        </el-form-item>
      </el-col>
    </el-row>

    <el-alert
      v-if="selectedPreset?.deferred"
      type="warning"
      title="This preset is not available in the MVP build"
      show-icon
      :closable="false"
      class="mb-12"
    />

    <el-row v-if="showAdvanced && local.enabled" :gutter="16" class="advanced-row">
      <el-col :xs="24" :sm="12">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>Seed mode</span>
              <FieldHelpIcon :field="fieldFor('seed_mode')" />
            </span>
          </template>
          <el-select
            :model-value="local.seed_mode || 'deterministic_per_image'"
            class="field-full"
            @update:model-value="(v) => patch({ seed_mode: String(v) })"
          >
            <el-option
              v-for="mode in catalog?.seed_modes || ['deterministic_per_image']"
              :key="mode"
              :label="mode"
              :value="mode"
            />
          </el-select>
        </el-form-item>
      </el-col>
      <el-col :xs="24" :sm="12">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>Variant sampling</span>
              <FieldHelpIcon :field="fieldFor('variant_sampling')" />
            </span>
          </template>
          <el-select
            :model-value="local.variant_sampling || 'probability'"
            class="field-full"
            @update:model-value="(v) => patch({ variant_sampling: String(v) })"
          >
            <el-option
              v-for="mode in catalog?.variant_sampling_modes || ['probability', 'enumerated']"
              :key="mode"
              :label="mode"
              :value="mode"
            />
          </el-select>
        </el-form-item>
      </el-col>
      <el-col :xs="24" :sm="12">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>Max branches per image</span>
              <FieldHelpIcon :field="fieldFor('max_branches_per_image')" />
            </span>
          </template>
          <el-input-number
            :model-value="numOrNull(local.max_branches_per_image)"
            :min="1"
            :step="1"
            class="field-narrow"
            @update:model-value="(v) => patch({ max_branches_per_image: v == null ? undefined : Number(v) })"
          />
        </el-form-item>
      </el-col>
    </el-row>

    <div v-if="showStrategies && (local.enabled || hideEnable)" class="strategies-block">
      <div class="strategies-head">
        <span class="strategies-title">Strategy overrides</span>
        <FieldHelpIcon :field="fieldFor('strategies')" />
        <el-select
          ref="addStrategySelectRef"
          v-model="addStrategyName"
          placeholder="Add strategy…"
          clearable
          filterable
          class="add-strategy-select"
          @change="onAddStrategy"
        >
          <el-option
            v-for="strategy in addableStrategies"
            :key="strategy.name"
            :label="strategy.label"
            :value="strategy.name"
          />
        </el-select>
      </div>
      <p class="strategies-hint">
        Override preset defaults per strategy. Preset
        <strong>{{ selectedPreset?.label || local.preset }}</strong>
        includes:
        {{ presetStrategyNames.length ? presetStrategyNames.join(", ") : "none" }}.
      </p>

      <el-empty
        v-if="!strategyOverrideNames.length"
        description="No strategy overrides — preset defaults only"
        :image-size="48"
      >
        <el-button
          type="primary"
          :icon="Plus"
          :disabled="!addableStrategies.length"
          @click="openAddStrategyPicker"
        >
          Add strategy override
        </el-button>
      </el-empty>

      <el-collapse v-else v-model="openStrategies" class="strategy-collapse">
        <el-collapse-item
          v-for="name in strategyOverrideNames"
          :key="name"
          :name="name"
        >
          <template #title>
            <span class="strategy-collapse-title">
              <span>{{ strategyLabel(name) }}</span>
              <FieldHelpIcon
                v-if="strategyHelpField(name)"
                :field="strategyHelpField(name)!"
                @click.stop
              />
            </span>
            <el-button
              type="danger"
              link
              size="small"
              class="remove-strategy"
              @click.stop="removeStrategy(name)"
            >
              Remove
            </el-button>
          </template>
          <StrategyOverrideEditor
            :strategy="strategyMeta(name)"
            :params="local.strategies?.[name] || {}"
            @update="(params) => updateStrategy(name, params)"
          />
        </el-collapse-item>
      </el-collapse>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { Plus } from "@element-plus/icons-vue";
import type { ElSelect } from "element-plus";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import {
  availablePresets,
  emptyAugmentationConfig,
  implementedStrategies,
  lookupSchemaField,
  mergeAugmentationPatch,
  schemaFieldsByPath,
  setStrategyOverride,
  type AugmentationCatalog,
  type AugmentationConfig,
  type AugStrategyCatalogEntry,
} from "../lib/datasetAugmentation";
import type { SchemaField } from "../types/forms";
import StrategyOverrideEditor from "./StrategyOverrideEditor.vue";

const props = withDefaults(
  defineProps<{
    config: AugmentationConfig;
    catalog: AugmentationCatalog | null;
    schemaFields?: SchemaField[];
    showAdvanced?: boolean;
    showStrategies?: boolean;
    disabled?: boolean;
    hideEnable?: boolean;
    hidePreset?: boolean;
  }>(),
  {
    schemaFields: () => [],
    showAdvanced: false,
    showStrategies: false,
    disabled: false,
    hideEnable: false,
    hidePreset: false,
  }
);

const emit = defineEmits<{
  (e: "update", value: AugmentationConfig): void;
}>();

const local = ref<AugmentationConfig>({ ...emptyAugmentationConfig() });
const addStrategyName = ref("");
const addStrategySelectRef = ref<InstanceType<typeof ElSelect> | null>(null);
const openStrategies = ref<string[]>([]);

const fieldsByPath = computed(() => schemaFieldsByPath(props.schemaFields));

function fieldFor(path: string): SchemaField {
  const labels: Record<string, string> = {
    enabled: "Enable augmentation",
    preset: "Preset",
    seed_mode: "Seed mode",
    variant_sampling: "Variant sampling",
    max_branches_per_image: "Max branches per image",
    strategies: "Strategy overrides",
  };
  return lookupSchemaField(fieldsByPath.value, path, labels[path] || path);
}

watch(
  () => props.config,
  (value) => {
    local.value = { ...emptyAugmentationConfig(), ...value };
    if (value.strategies) {
      local.value.strategies = { ...value.strategies };
    }
  },
  { immediate: true, deep: true }
);

const presetOptions = computed(() => availablePresets(props.catalog));

const selectedPreset = computed(() =>
  props.catalog?.presets?.find((p) => p.name === (local.value.preset || "none"))
);

const presetStrategyNames = computed(() => selectedPreset.value?.strategies ?? []);

const strategyOverrideNames = computed(() =>
  Object.keys(local.value.strategies ?? {}).sort()
);

const addableStrategies = computed(() =>
  implementedStrategies(props.catalog).filter(
    (s) => !local.value.strategies?.[s.name]
  )
);

function numOrNull(value: unknown): number | null {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function patch(partial: Partial<AugmentationConfig>) {
  if (props.disabled) return;
  const next = mergeAugmentationPatch(local.value, partial);
  local.value = next;
  emit("update", next);
}

function strategyMeta(name: string): AugStrategyCatalogEntry | undefined {
  return props.catalog?.strategies?.find((s) => s.name === name);
}

function strategyLabel(name: string): string {
  return strategyMeta(name)?.label || name.replace(/_/g, " ");
}

function strategyHelpField(name: string): SchemaField | null {
  const meta = strategyMeta(name);
  const help = meta?.help?.trim();
  if (!help) return null;
  return {
    path: `augmentation.strategy.${name}`,
    label: strategyLabel(name),
    type: "string",
    help,
    doc_path: "docs/user/dataset-augmentation.md",
  };
}

function openAddStrategyPicker(): void {
  addStrategySelectRef.value?.focus?.();
}

function onAddStrategy(name: string | null | undefined) {
  if (!name) return;
  const meta = strategyMeta(name);
  const defaults: Record<string, unknown> = { enabled: true };
  for (const field of meta?.parameters ?? []) {
    if (field.default !== undefined) defaults[field.path] = field.default;
  }
  const next = setStrategyOverride(local.value, name, defaults);
  local.value = next;
  openStrategies.value = [...openStrategies.value, name];
  addStrategyName.value = "";
  emit("update", next);
}

function updateStrategy(name: string, params: Record<string, unknown>) {
  const next = setStrategyOverride(local.value, name, params);
  local.value = next;
  emit("update", next);
}

function removeStrategy(name: string) {
  const next = setStrategyOverride(local.value, name, null);
  local.value = next;
  openStrategies.value = openStrategies.value.filter((n) => n !== name);
  emit("update", next);
}
</script>

<style scoped>
.augmentation-config-editor :deep(.el-form-item) {
  margin-bottom: 12px;
}
.label-row {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
.mb-12 {
  margin-bottom: 12px;
}
.advanced-row {
  margin-bottom: 4px;
}
.strategies-block {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--el-border-color-lighter);
}
.strategies-head {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.strategies-title {
  font-weight: 600;
}
.add-strategy-select {
  min-width: 180px;
  margin-left: auto;
}
.strategies-hint {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.strategy-collapse :deep(.el-collapse-item__header) {
  font-size: 13px;
}
.strategy-collapse-title {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.remove-strategy {
  margin-left: auto;
  margin-right: 8px;
}
</style>

<template>
  <div class="resolution-schedule-field">
    <template v-if="!showJson">
      <div class="enable-row">
        <el-switch :model-value="enabled" @update:model-value="(v) => setEnabled(Boolean(v))" />
        <el-text size="small" :type="enabled ? 'primary' : 'info'">
          {{ enabled ? "Schedule active" : "Schedule off (resolutions mixed uniformly)" }}
        </el-text>
      </div>

      <el-alert
        v-if="!availableResolutions.length"
        type="info"
        :closable="false"
        show-icon
        class="hint-alert"
        title="Add resolutions above first"
        description="Stages pick from the dataset's Resolutions list."
      />

      <template v-else>
        <el-empty
          v-if="!rows.length"
          description="No stages yet — the whole run mixes all resolutions"
          :image-size="48"
        >
          <el-button type="primary" :icon="Plus" @click="addStage">Add stage</el-button>
        </el-empty>

        <div v-for="(row, idx) in rows" :key="idx" class="stage-row">
          <span class="stage-index">{{ idx + 1 }}</span>
          <el-select
            :model-value="row.resolutions"
            multiple
            placeholder="Resolutions"
            class="stage-res"
            @update:model-value="(v) => updateResolutions(idx, v as number[])"
          >
            <el-option
              v-for="opt in resolutionOptions(row)"
              :key="opt"
              :label="String(opt)"
              :value="opt"
            />
          </el-select>
          <el-input-number
            :model-value="row.fraction"
            :min="0.01"
            :step="0.05"
            :precision="2"
            controls-position="right"
            class="stage-fraction"
            @update:model-value="(v) => updateFraction(idx, v)"
          />
          <el-tag size="small" type="info" class="stage-percent">{{ percentLabel(idx) }}</el-tag>
          <el-button type="danger" link :icon="Delete" @click="removeStage(idx)" />
        </div>

        <el-space v-if="rows.length" wrap class="actions-row">
          <el-button size="small" :icon="Plus" @click="addStage">Add stage</el-button>
          <el-button size="small" link @click="showJson = true">Edit as JSON</el-button>
        </el-space>

        <el-text v-if="rows.length" type="info" size="small" class="explain">
          Stages run in order. Fractions are normalized to 100% (so [1, 1, 2] = 25% / 25% / 50%).
          One resolution per stage = staged (no mixing); several = mixed during that stage.
        </el-text>
        <el-text v-if="unknownResolutionWarning" type="warning" size="small" class="explain">
          {{ unknownResolutionWarning }}
        </el-text>
      </template>
    </template>

    <template v-else>
      <el-input
        :model-value="jsonText"
        type="textarea"
        :rows="6"
        class="field-full"
        placeholder='{ "enabled": true, "stage": [ { "resolutions": [512], "fraction": 0.33 } ] }'
        @update:model-value="onJsonInput"
      />
      <el-text v-if="jsonError" type="danger" size="small" class="json-error">{{ jsonError }}</el-text>
      <el-button
        v-if="canUseTableEditor"
        size="small"
        link
        class="mt-8"
        @click="showJson = false"
      >
        Back to table editor
      </el-button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, type PropType } from "vue";
import { Delete, Plus } from "@element-plus/icons-vue";
import { jsonStringify } from "../lib/formUtils";
import {
  parseResolutionSchedule,
  scheduleFormValue,
  scheduleNeedsJsonEditor,
  stageEffectivePercent,
  validateResolutionScheduleJson,
  type ResolutionSchedule,
  type ScheduleStage,
} from "../lib/resolutionSchedule";

const props = defineProps({
  modelValue: {
    type: [String, Object] as PropType<string | Record<string, unknown> | null>,
    default: "",
  },
  availableResolutions: {
    type: Array as PropType<number[]>,
    default: () => [],
  },
});

const emit = defineEmits(["update:modelValue"]);

const showJson = ref(false);

watch(
  () => props.modelValue,
  (value) => {
    if (scheduleNeedsJsonEditor(value)) showJson.value = true;
  },
  { immediate: true }
);

const schedule = computed(() => parseResolutionSchedule(props.modelValue));
const enabled = computed(() => schedule.value.enabled);
const rows = computed(() => schedule.value.stages);

const jsonText = computed(() => jsonStringify(props.modelValue));
const jsonError = computed(() =>
  showJson.value ? validateResolutionScheduleJson(jsonText.value) : null
);
const canUseTableEditor = computed(() => !jsonError.value);

const percents = computed(() => stageEffectivePercent(rows.value));

function percentLabel(index: number): string {
  return `${Math.round(percents.value[index] ?? 0)}%`;
}

function resolutionOptions(row: ScheduleStage): number[] {
  // Offer the dataset's resolutions plus any already selected (even if removed upstream).
  const set = new Set<number>(props.availableResolutions);
  for (const r of row.resolutions) set.add(r);
  return Array.from(set).sort((a, b) => a - b);
}

const unknownResolutionWarning = computed(() => {
  const known = new Set(props.availableResolutions);
  const unknown = new Set<number>();
  for (const row of rows.value) {
    for (const r of row.resolutions) if (!known.has(r)) unknown.add(r);
  }
  if (!unknown.size) return "";
  return `Resolution(s) ${Array.from(unknown).sort((a, b) => a - b).join(", ")} are not in the dataset's Resolutions list — they won't be cached.`;
});

function emitSchedule(next: ResolutionSchedule): void {
  emit("update:modelValue", scheduleFormValue(next));
}

function setEnabled(value: boolean): void {
  emitSchedule({ enabled: value, stages: rows.value });
}

function addStage(): void {
  const first = props.availableResolutions[0];
  if (first === undefined) return;
  emitSchedule({
    enabled: true,
    stages: [...rows.value, { resolutions: [first], fraction: 1 }],
  });
}

function removeStage(index: number): void {
  emitSchedule({ enabled: enabled.value, stages: rows.value.filter((_, i) => i !== index) });
}

function updateResolutions(index: number, value: number[]): void {
  const cleaned = value.map(Number).filter((n) => Number.isFinite(n) && n > 0);
  // Keep at least one resolution; use the "remove" button to drop a stage.
  if (cleaned.length === 0) return;
  const stages = rows.value.map((row, i) =>
    i === index ? { ...row, resolutions: cleaned } : row
  );
  emitSchedule({ enabled: enabled.value, stages });
}

function updateFraction(index: number, value: number | undefined): void {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return;
  const stages = rows.value.map((row, i) => (i === index ? { ...row, fraction: n } : row));
  emitSchedule({ enabled: enabled.value, stages });
}

function onJsonInput(text: string): void {
  const trimmed = text.trim();
  emit("update:modelValue", trimmed ? text : "");
}
</script>

<style scoped>
.resolution-schedule-field {
  width: 100%;
}
.enable-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.hint-alert {
  margin-bottom: 8px;
}
.stage-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.stage-index {
  width: 18px;
  text-align: center;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.stage-res {
  min-width: 220px;
  flex: 1 1 220px;
}
.stage-fraction {
  width: 120px;
}
.stage-percent {
  min-width: 44px;
  text-align: center;
}
.actions-row {
  margin-top: 4px;
}
.explain {
  display: block;
  margin-top: 8px;
  line-height: 1.5;
}
.json-error {
  display: block;
  margin-top: 6px;
}
.mt-8 {
  margin-top: 8px;
}
.field-full {
  width: 100%;
}
</style>

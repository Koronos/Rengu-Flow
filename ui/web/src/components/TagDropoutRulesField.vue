<template>
  <div class="tag-dropout-rules-field">
    <template v-if="!showJson">
      <el-empty v-if="!rows.length" description="No tag dropout rules yet" :image-size="48">
        <el-button type="primary" :icon="Plus" @click="addRow">Add tag dropout rule</el-button>
      </el-empty>

      <div v-for="(row, idx) in rows" :key="idx" class="rule-row">
        <div class="rule-row-head">
          <el-radio-group
            :model-value="row.source"
            size="small"
            @update:model-value="(v) => setSource(idx, v as TagDropoutRuleSource)"
          >
            <el-radio-button value="tags">Tag list</el-radio-button>
            <el-radio-button value="file">Tags file (.txt)</el-radio-button>
          </el-radio-group>
          <el-button type="danger" link :icon="Delete" @click="removeRow(idx)" />
        </div>

        <el-select
          v-if="row.source === 'tags'"
          :model-value="row.tags"
          multiple
          filterable
          allow-create
          default-first-option
          collapse-tags
          collapse-tags-tooltip
          :max-collapse-tags="4"
          placeholder="Type a tag and press Enter"
          class="field-full tags-select"
          @update:model-value="(v) => updateRow(idx, { tags: normalizeTags(v) })"
        />

        <PathFieldControl
          v-else
          :model-value="row.tags_file"
          expect="file"
          placeholder="Path to .txt file (one tag per line)"
          input-class="field-path field-full"
          @update:model-value="(v) => updateRow(idx, { tags_file: v })"
        />

        <div class="prob-row">
          <span class="prob-label">Drop probability</span>
          <el-slider
            :model-value="row.drop_probability"
            :min="0"
            :max="1"
            :step="0.01"
            :show-tooltip="true"
            :format-tooltip="formatProbTooltip"
            class="prob-slider"
            @update:model-value="(v) => updateProbability(idx, v)"
          />
          <el-input-number
            :model-value="row.drop_probability"
            :min="0"
            :max="1"
            :step="0.05"
            :precision="2"
            controls-position="right"
            class="field-narrow prob-input"
            @update:model-value="(v) => updateProbability(idx, v)"
          />
        </div>
      </div>

      <el-space v-if="rows.length" wrap class="actions-row">
        <el-button size="small" :icon="Plus" @click="addRow">Add tag dropout rule</el-button>
        <el-button size="small" link @click="showJson = true">Edit as JSON</el-button>
      </el-space>
    </template>

    <template v-else>
      <el-input
        :model-value="jsonText"
        type="textarea"
        :rows="6"
        class="field-full"
        placeholder='[{"tags":["char"],"drop_probability":0.08},{"tags_file":"drop.txt","drop_probability":0.5}]'
        @update:model-value="onJsonInput"
      />
      <el-text v-if="jsonError" type="danger" size="small" class="json-error">
        {{ jsonError }}
      </el-text>
      <el-button
        v-if="canUseTableEditor"
        size="small"
        link
        class="mt-8"
        @click="showJson = false"
      >
        Back to rule editor
      </el-button>
    </template>

    <p v-if="hint" class="field-hint">{{ hint }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, type PropType } from "vue";
import { Delete, Plus } from "@element-plus/icons-vue";
import PathFieldControl from "./PathFieldControl.vue";
import { jsonStringify } from "../lib/formUtils";
import {
  emptyTagDropoutRule,
  parseTagDropoutRules,
  tagDropoutRulesFormValue,
  tagDropoutRulesNeedJsonEditor,
  validateTagDropoutRulesJson,
  type TagDropoutRuleSource,
  type TagDropoutRuleUi,
} from "../lib/tagDropoutRules";

const props = defineProps({
  modelValue: {
    type: [Array, String] as PropType<unknown[] | string | null>,
    default: () => [],
  },
  hint: { type: String, default: "" },
});

const emit = defineEmits(["update:modelValue"]);

const showJson = ref(false);

watch(
  () => props.modelValue,
  (value) => {
    if (tagDropoutRulesNeedJsonEditor(value)) {
      showJson.value = true;
    }
  },
  { immediate: true }
);

const rows = computed(() => parseTagDropoutRules(props.modelValue));

const jsonText = computed(() => jsonStringify(props.modelValue));

const jsonError = computed(() => {
  if (!showJson.value) return null;
  return validateTagDropoutRulesJson(jsonText.value);
});

const canUseTableEditor = computed(() => !jsonError.value && jsonText.value.trim() !== "");

function emitRows(next: TagDropoutRuleUi[]): void {
  emit("update:modelValue", tagDropoutRulesFormValue(next));
}

function addRow(): void {
  emitRows([...rows.value, emptyTagDropoutRule()]);
}

function removeRow(index: number): void {
  emitRows(rows.value.filter((_, i) => i !== index));
}

function updateRow(index: number, patch: Partial<TagDropoutRuleUi>): void {
  const next = rows.value.map((row, i) => (i === index ? { ...row, ...patch } : row));
  emitRows(next);
}

function setSource(index: number, source: TagDropoutRuleSource): void {
  updateRow(index, { source });
}

function normalizeTags(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((t) => String(t).trim()).filter(Boolean);
}

function updateProbability(index: number, value: number | undefined): void {
  const n = Number(value);
  if (!Number.isFinite(n)) return;
  updateRow(index, { drop_probability: Math.min(1, Math.max(0, n)) });
}

function formatProbTooltip(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function onJsonInput(text: string): void {
  const trimmed = text.trim();
  emit("update:modelValue", trimmed ? text : "");
}
</script>

<style scoped>
.tag-dropout-rules-field {
  width: 100%;
}
.rule-row {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
.rule-row-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.tags-select {
  margin-bottom: 8px;
}
.prob-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
  margin-top: 4px;
}
.prob-label {
  font-size: 13px;
  color: var(--el-text-color-secondary);
  flex-shrink: 0;
}
.prob-slider {
  flex: 1;
  min-width: 120px;
  max-width: 280px;
}
.actions-row {
  margin-top: 4px;
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
.field-hint {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
</style>

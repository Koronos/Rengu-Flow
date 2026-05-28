<template>
  <div class="dataset-tag-dropout-field">
    <el-row :gutter="16">
      <el-col :xs="24" :sm="12">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>{{ labelFor("tag_dropout_enabled") }}</span>
              <FieldHelpIcon :field="fieldFor('tag_dropout_enabled')" />
            </span>
          </template>
          <el-switch
            :model-value="Boolean(form.tag_dropout_enabled)"
            @update:model-value="(v) => emit('update:path', 'tag_dropout_enabled', v)"
          />
        </el-form-item>
      </el-col>
      <el-col :xs="24" :sm="12">
        <el-form-item>
          <template #label>
            <span class="label-row">
              <span>{{ labelFor("uncond_fraction") }}</span>
              <FieldHelpIcon :field="fieldFor('uncond_fraction')" />
            </span>
          </template>
          <el-input-number
            :model-value="num(form.uncond_fraction)"
            :min="0"
            :max="1"
            :step="0.05"
            class="field-narrow"
            @update:model-value="(v) => emit('update:path', 'uncond_fraction', v)"
          />
        </el-form-item>
      </el-col>
    </el-row>
    <template v-if="form.tag_dropout_enabled">
      <el-row :gutter="16">
        <el-col :xs="24" :sm="8">
          <el-form-item>
            <template #label>
              <span class="label-row">
                <span>{{ labelFor("tag_dropout_probability") }}</span>
                <FieldHelpIcon :field="fieldFor('tag_dropout_probability')" />
              </span>
            </template>
            <el-input-number
              :model-value="num(form.tag_dropout_probability)"
              :min="0"
              :max="1"
              :step="0.05"
              class="field-narrow"
              @update:model-value="(v) => emit('update:path', 'tag_dropout_probability', v)"
            />
          </el-form-item>
        </el-col>
        <el-col :xs="24" :sm="8">
          <el-form-item>
            <template #label>
              <span class="label-row">
                <span>{{ labelFor("tag_dropout_mode") }}</span>
                <FieldHelpIcon :field="fieldFor('tag_dropout_mode')" />
              </span>
            </template>
            <el-select
              :model-value="String(form.tag_dropout_mode || 'per_tag')"
              @update:model-value="(v) => emit('update:path', 'tag_dropout_mode', v)"
            >
              <el-option label="per_tag" value="per_tag" />
              <el-option label="full" value="full" />
            </el-select>
          </el-form-item>
        </el-col>
        <el-col :xs="24" :sm="8">
          <el-form-item>
            <template #label>
              <span class="label-row">
                <span>{{ labelFor("tag_match_case_sensitive") }}</span>
                <FieldHelpIcon :field="fieldFor('tag_match_case_sensitive')" />
              </span>
            </template>
            <el-switch
              :model-value="Boolean(form.tag_match_case_sensitive)"
              @update:model-value="(v) => emit('update:path', 'tag_match_case_sensitive', v)"
            />
          </el-form-item>
        </el-col>
      </el-row>
      <el-form-item>
        <template #label>
          <span class="label-row">
            <span>{{ labelFor("tag_dropout_rules") }}</span>
            <FieldHelpIcon :field="fieldFor('tag_dropout_rules')" />
          </span>
        </template>
        <TagDropoutRulesField
          :model-value="form.tag_dropout_rules"
          :hint="rulesHint"
          @update:model-value="(v) => emit('update:path', 'tag_dropout_rules', v)"
        />
      </el-form-item>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import FieldHelpIcon from "./FieldHelpIcon.vue";
import TagDropoutRulesField from "./TagDropoutRulesField.vue";
import type { FormValues, SchemaField } from "../types/forms";

const props = defineProps<{
  form: FormValues;
  schemaFields: SchemaField[];
}>();

const emit = defineEmits<{
  "update:path": [path: string, value: unknown];
}>();

const fieldsByPath = computed(() => {
  const map = new Map<string, SchemaField>();
  for (const f of props.schemaFields) {
    if (f.path) map.set(f.path, f);
  }
  return map;
});

function fieldFor(path: string): SchemaField {
  return (
    fieldsByPath.value.get(path) ?? {
      path,
      label: path,
      type: "string",
      help: path,
    }
  );
}

function labelFor(path: string): string {
  return fieldFor(path).label || path;
}

function num(v: unknown): number {
  if (v === "" || v === null || v === undefined) return 0;
  return Number(v);
}

const rulesHint = computed(() => {
  const help = fieldFor("tag_dropout_rules").help;
  if (typeof help === "string" && help.trim()) return help;
  return (
    "Override drop probability for specific tags. Tags not listed use the default probability above. " +
    "Each rule is either an inline tag list or a .txt file with one tag per line."
  );
});
</script>

<style scoped>
.label-row {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
</style>

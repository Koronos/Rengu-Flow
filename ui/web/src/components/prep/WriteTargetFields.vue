<!--
  Where a tag / caption / edit-instruction step writes, and what it does when the line is taken:
  the target line (any 1-based number) and the write mode (skip / replace / append).

  One component for the three stage forms (and so for the workflow drawer and the standalone prep
  job form alike). It edits the stage form in place: `write_mode` is stored only once the user
  picks one, so a form that was merely opened keeps producing the config it was seeded from.
  Until then the shown mode is derived from the legacy `overwrite` switch (on = replace), which is
  also what the server does with a config that has no `write_mode`.
-->
<template>
  <div class="form-row-2">
    <el-form-item>
      <template #label>
        {{ lineLabel }} <FieldHelpIcon :field="help(lineHelp)" />
        <FieldPathTag :path="`${stage}.target_line`" />
      </template>
      <el-input-number
        v-model="targetLine"
        :min="1"
        :step="1"
        :precision="0"
        :placeholder="String(defaultLine)"
        controls-position="right"
      />
      <el-text size="small" type="info" class="hint-text">{{ lineHint }}</el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        If the line already has text
        <FieldHelpIcon
          :field="help('Skip leaves images whose target line already has text untouched, so a stopped job resumes where it left off (default). Replace overwrites that line. Append adds to the end of it, ' + joinHint + '. An empty line is simply written.')"
        />
        <FieldPathTag :path="`${stage}.write_mode`" />
      </template>
      <el-select v-model="mode" class="w-full">
        <el-option label="Skip it (leave as is)" value="skip" />
        <el-option label="Replace it" value="replace" />
        <el-option label="Append to it" value="append" />
      </el-select>
    </el-form-item>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PropType } from "vue";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import { help } from "./formHelpers";
import type { WriteMode } from "../../lib/prepStageConfig";

interface WriteTargetForm {
  target_line: number;
  write_mode: WriteMode;
  overwrite: boolean;
}

const props = defineProps({
  /** The stage form, edited in place (the parents' `v-model` object). */
  form: { type: Object as PropType<WriteTargetForm>, required: true },
  stage: { type: String as PropType<"tag" | "caption" | "edit_caption">, required: true },
});

const defaultLine = computed(() => (props.stage === "caption" ? 2 : 1));

const lineLabel = computed(() => (props.stage === "tag" ? "Target line" : "Caption line"));

const lineHelp = computed(() => {
  if (props.stage === "tag") {
    return "1-based caption line the tags are written to (default 1). Any line works; if the caption has fewer lines the ones before it are padded with empty lines (line 3 on an uncaptioned image leaves lines 1 and 2 empty).";
  }
  if (props.stage === "caption") {
    return "1-based caption line the caption is written to (default 2, below the tag line). Any line works, including line 1; a shorter caption is padded with empty lines first. Every non-empty line is one caption variant at training time.";
  }
  return "1-based caption line the edit instruction is written to (default 1, the line an edit dataset trains on). A shorter caption is padded with empty lines first.";
});

const lineHint = computed(() =>
  props.stage === "tag"
    ? "Line 1 = the tag line. Shorter captions are padded with empty lines."
    : props.stage === "caption"
      ? "Line 2 = the standard caption. 3+ adds a variant; shorter captions are padded."
      : "Line 1 = what an edit dataset trains on. Shorter captions are padded.",
);

const joinHint = computed(() =>
  props.stage === "tag"
    ? "joined with a comma and without repeating a tag"
    : "joined with a single space",
);

const targetLine = computed<number>({
  get: () => props.form.target_line,
  set: (value) => {
    props.form.target_line = Number.isFinite(value) && value >= 1 ? Math.trunc(value) : defaultLine.value;
  },
});

const mode = computed<"skip" | "replace" | "append">({
  get: () => props.form.write_mode || (props.form.overwrite ? "replace" : "skip"),
  set: (value) => {
    props.form.write_mode = value;
    // Kept in step so a reader of the old key (and an older server) still sees the same intent.
    props.form.overwrite = value === "replace";
  },
});
</script>

<style scoped>
.form-row-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 600px) {
  .form-row-2 {
    grid-template-columns: 1fr;
  }
}
.hint-text {
  display: block;
  margin-top: 4px;
  line-height: 1.4;
}
</style>

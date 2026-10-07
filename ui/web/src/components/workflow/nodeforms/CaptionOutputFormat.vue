<!--
  Where a tag / caption / edit-instruction step keeps its captions: as sidecar files (one .txt per
  image) or as a single captions.json. The default is "Same as input" - the step reads and writes
  the layout it receives, which is what every workflow saved before this control did.

  Picking the other layout makes the step **convert** the folder first (new files written, then the
  old ones removed, so training - where a captions.json wins - never reads a stale copy), run in
  the new layout, and hand it to the steps below. So a chain that starts with a tool, with no
  source folder to hold the setting, still gets a place to choose.

  The control is the node's own (`output_format` / `output_ext` in its config), not the edge's: it
  edits only this step and reports the choice upward as a plain `{ format, ext }`.
-->
<template>
  <div class="caption-output">
    <div class="caption-output__row">
      <el-text size="small" class="caption-output__label">Caption format</el-text>
      <el-select
        v-model="format"
        size="small"
        class="caption-output__select"
        :disabled="disabled"
        v-bind="ariaLabel('Caption format')"
      >
        <el-option value="inherit" :label="`Same as input (${inheritedLabel})`" />
        <el-option value="sidecar" label="Sidecar files (one text file per image)" />
        <el-option value="json" label="captions.json (one file for the folder)" />
      </el-select>
      <el-input
        v-if="format === 'sidecar'"
        v-model="ext"
        size="small"
        class="caption-output__ext"
        placeholder=".txt"
        :disabled="disabled"
        v-bind="ariaLabel('Sidecar extension')"
        @blur="normalizeExt"
      />
      <el-button
        v-if="format === 'inherit' && showChangeLink"
        size="small"
        link
        type="primary"
        @click="emit('open-origin')"
      >
        Change format
      </el-button>
    </div>
    <el-text size="small" type="info" class="caption-output__hint">
      <template v-if="format === 'inherit'">
        Captions: <strong>{{ inheritedLabel }}</strong>
        <template v-if="originText"> · set on {{ originText }}</template>
      </template>
      <template v-else>
        Captions already in the folder are converted to <strong>{{ chosenLabel }}</strong> before
        this step runs, and the old files are removed so training never reads a stale copy. The
        steps below read this layout.
      </template>
    </el-text>
    <el-text
      v-if="format === 'sidecar' && ext.trim() && ext.trim().toLowerCase().replace(/^\./, '') !== 'txt'"
      size="small"
      type="warning"
      class="caption-output__hint caption-output__warn"
    >
      Training reads only <code>.txt</code> sidecars: with <code>{{ ext }}</code> the captions are
      not seen by a training run. Use <code>.txt</code> unless something else reads these files.
    </el-text>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { ariaLabel } from "../../../lib/aria";

export interface CaptionOutputValue {
  format: "inherit" | "sidecar" | "json";
  ext: string;
}

const model = defineModel<CaptionOutputValue>({ required: true });

defineProps({
  /** The layout the step receives, as a phrase: `sidecar files (.txt)` / `captions.json`. */
  inheritedLabel: { type: String, default: "sidecar files (.txt)" },
  /** The step that sets the incoming layout, e.g. `① Source folder`; `""` when unknown. */
  originText: { type: String, default: "" },
  /** Offer the shortcut to the source folder step that owns the incoming layout. */
  showChangeLink: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits<{ (e: "open-origin"): void }>();

const format = computed<CaptionOutputValue["format"]>({
  get: () => model.value.format,
  set: (value) => {
    model.value = { ...model.value, format: value, ext: model.value.ext || ".txt" };
  },
});

const ext = computed<string>({
  get: () => model.value.ext,
  set: (value) => {
    model.value = { ...model.value, ext: value };
  },
});

function normalizeExt(): void {
  const trimmed = model.value.ext.trim();
  const next = trimmed ? (trimmed.startsWith(".") ? trimmed : `.${trimmed}`) : ".txt";
  if (next !== model.value.ext) model.value = { ...model.value, ext: next };
}

const chosenLabel = computed(() =>
  model.value.format === "json" ? "captions.json" : `sidecar files (${model.value.ext || ".txt"})`,
);

</script>

<style scoped>
.caption-output__row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.caption-output__select {
  width: 300px;
  max-width: 100%;
}
.caption-output__ext {
  width: 96px;
}
.caption-output__hint {
  display: block;
  margin-top: 4px;
  line-height: 1.45;
}
</style>

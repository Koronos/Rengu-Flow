<!--
  Dataset-level fields shared by every prep stage: the folder plus the caption
  layout. In a workflow these three come from the incoming edge (the executor
  injects them from the handle), so the whole block is hidden with
  `hide-dataset-fields`.
-->
<template>
  <el-form v-if="!hideDatasetFields" label-position="top" :disabled="disabled">
    <el-form-item required>
      <template #label>
        Dataset folder <FieldHelpIcon :field="help('Path to the image folder to process. Only images directly inside the folder are scanned (no subfolders — same rule as training).')" />
        <FieldPathTag path="path" />
      </template>
      <PathFieldControl
        v-model="model.path"
        expect="dir"
        required
        placeholder="e.g. /path/to/dataset"
        input-class="w-full"
      />
    </el-form-item>

    <!-- Cleanup, quality filter and the quality index read images only — caption
         layout is irrelevant to them. -->
    <div v-if="showCaptionLayout" class="form-row-2">
      <el-form-item>
        <template #label>
          Caption format <FieldHelpIcon :field="help('Sidecar: one .txt per image, each line a caption variant (line 1 = tags, line 2 = caption). JSON: single captions.json index file per folder.')" />
          <FieldPathTag path="caption_format" />
        </template>
        <el-select v-model="model.caption_format" class="w-full">
          <el-option label="Caption files (.txt next to each image)" value="sidecar" />
          <el-option label="captions.json (single index file)" value="json" />
        </el-select>
      </el-form-item>
      <el-form-item v-if="model.caption_format !== 'json'">
        <template #label>
          Caption extension <FieldHelpIcon :field="help('Extension of the per-image sidecar files read and written (default .txt). Change it only if your trainer or downstream tooling expects a different one.')" />
          <FieldPathTag path="caption_ext" />
        </template>
        <el-input v-model="model.caption_ext" placeholder=".txt" class="w-full" />
      </el-form-item>
    </div>
  </el-form>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PropType } from "vue";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import PathFieldControl from "../PathFieldControl.vue";
import { help } from "./formHelpers";
import type { PrepCommonForm } from "../../lib/prepStageConfig";
import type { PrepStage } from "../../types/api";

const model = defineModel<PrepCommonForm>({ required: true });

const props = defineProps({
  stage: { type: String as PropType<PrepStage>, required: true },
  /** Hide the whole block: a workflow node inherits these from its input edge. */
  hideDatasetFields: { type: Boolean, default: false },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the block — including the path picker's own button.
   */
  disabled: { type: Boolean, default: false },
});

const NO_CAPTION_LAYOUT: PrepStage[] = ["clean", "quality", "index"];
const showCaptionLayout = computed(() => !NO_CAPTION_LAYOUT.includes(props.stage));
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
</style>

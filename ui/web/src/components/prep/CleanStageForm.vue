<!-- Watermark cleanup options, extracted verbatim from PrepJobFormView.vue. -->
<template>
  <h3 class="section-title">Cleaning options</h3>
  <el-alert
    type="info"
    :closable="false"
    show-icon
    class="mt-8 mb-12"
    title="Watermark cleanup: a YOLO11 detector finds watermarks and signatures, then LaMa inpainting removes them. Non-destructive by default — cleaned copies are written to the output folder."
  />
  <el-form label-position="top" :disabled="disabled">
    <el-form-item>
      <template #label>
        Confidence threshold <FieldHelpIcon :field="help('YOLO11 detection score required to flag a region as a watermark (default 0.35). Lower it if the detector keeps missing faint or small watermarks; raise it if clean areas are being incorrectly inpainted.')" />
        <FieldPathTag path="clean.confidence" />
      </template>
      <el-slider
        v-model="model.confidence"
        :min="0"
        :max="1"
        :step="0.01"
        show-input
        :show-input-controls="false"
      />
    </el-form-item>

    <el-form-item>
      <template #label>
        Mask dilation (px) <FieldHelpIcon :field="help('Expands each detected watermark region by this many pixels before inpainting (default 8). Increase it when inpainted areas show a leftover fringe around the original watermark edge.')" />
        <FieldPathTag path="clean.mask_dilation_px" />
      </template>
      <el-input-number v-model="model.mask_dilation_px" :min="0" :max="100" placeholder="8" controls-position="right" />
    </el-form-item>

    <el-form-item>
      <template #label>
        In-place <FieldHelpIcon :field="help('Overwrites originals rather than writing to a separate folder (originals are backed up under the app data dir first, not inside the dataset). Use it when you want the training folder itself to contain only cleaned images and do not need the side-by-side comparison.')" />
        <FieldPathTag path="clean.in_place" />
      </template>
      <el-switch v-model="model.in_place" />
      <el-text class="ml-8" size="small">In-place (overwrite originals)</el-text>
      <el-alert
        v-if="model.in_place"
        type="warning"
        show-icon
        :closable="false"
        class="mt-8"
        title="Originals are backed up under the app data dir before cleaning"
      />
    </el-form-item>

    <el-form-item v-if="!model.in_place">
      <template #label>
        Output directory <FieldHelpIcon :field="help('Where to write cleaned images. Defaults to &lt;dataset&gt;/cleaned/.')" />
        <FieldPathTag path="clean.output_dir" />
      </template>
      <PathFieldControl
        v-model="model.output_dir"
        expect="dir"
        placeholder="<dataset>/cleaned"
        input-class="w-full"
      />
    </el-form-item>

    <el-form-item>
      <template #label>
        Copy undetected images <FieldHelpIcon :field="help('Also copies images with no watermark detected into the output folder, so it ends up as a complete cleaned dataset. Turn off to write only the images that were actually inpainted.')" />
        <FieldPathTag path="clean.copy_undetected" />
      </template>
      <el-switch v-model="model.copy_undetected" />
      <el-text class="ml-8" size="small">Copy images with no detections to output</el-text>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { onMounted, watch } from "vue";
import type { PropType } from "vue";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import PathFieldControl from "../PathFieldControl.vue";
import { copyKnown, help } from "./formHelpers";
import type { PrepCleanForm } from "../../lib/prepStageConfig";
import type { PrepCleanConfig } from "../../types/api";

const model = defineModel<PrepCleanForm>({ required: true });

const props = defineProps({
  /** `clean` section of a cloned job config. */
  seed: { type: Object as PropType<PrepCleanConfig | null>, default: null },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the whole stage form.
   */
  disabled: { type: Boolean, default: false },
});

let seedApplied = false;

function applySeed(): void {
  if (!props.seed || seedApplied) return;
  seedApplied = true;
  copyKnown(model.value as unknown as Record<string, unknown>, props.seed);
}

watch(() => props.seed, applySeed);
onMounted(applySeed);
</script>

<style scoped>
.section-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.ml-8 {
  margin-left: 8px;
}
.mt-8 {
  margin-top: 8px;
}
</style>

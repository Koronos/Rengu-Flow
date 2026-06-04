<template>
  <div
    class="run-loss-monitor"
    :class="{
      'run-loss-monitor--loading': loading,
      'run-loss-monitor--loading-subtle': loading && !loadingStrong,
    }"
  >
    <el-row :gutter="16">
      <el-col :xs="24" :sm="12" class="monitor-col">
        <div class="monitor-panel">
          <el-tooltip :content="LOSS_PANEL_HINTS['train/epoch_loss']" placement="top" :show-after="300">
            <el-text type="info" size="small" class="panel-title panel-title--hint">
              Loss per epoch
            </el-text>
          </el-tooltip>
          <ScalarLineChart
            :scalars="scalars"
            tag="train/epoch_loss"
            x-axis-label="epoch"
            value-label="loss"
          />
        </div>
      </el-col>
      <el-col :xs="24" :sm="12" class="monitor-col">
        <div class="monitor-panel">
          <el-tooltip :content="LOSS_PANEL_HINTS['train/loss']" placement="top" :show-after="300">
            <el-text type="info" size="small" class="panel-title panel-title--hint">
              Loss per step
            </el-text>
          </el-tooltip>
          <ScalarLineChart
            :scalars="scalars"
            tag="train/loss"
            x-axis-label="step"
            value-label="loss"
          />
        </div>
      </el-col>
      <el-col :xs="24" :sm="12" class="monitor-col">
        <div class="monitor-panel">
          <el-tooltip :content="LOSS_PANEL_HINTS.step_jump" placement="top" :show-after="300">
            <el-text type="info" size="small" class="panel-title panel-title--hint">
              {{ stepJumpTitle }}
            </el-text>
          </el-tooltip>
          <ScalarLineChart
            :scalars="scalars"
            :tag="stepJumpTag"
            x-axis-label="step"
            :value-label="stepJumpValueLabel"
          />
        </div>
      </el-col>
    </el-row>
    <div class="monitor-panel monitor-panel--images">
      <PreviewStepBrowser :preview-images="previewImages" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PropType } from "vue";
import PreviewStepBrowser from "./PreviewStepBrowser.vue";
import ScalarLineChart from "./ScalarLineChart.vue";
import {
  LOSS_PANEL_HINTS,
  resolveStepJumpTag,
  stepJumpPanelTitle,
  type ScalarPoint,
} from "../lib/scalarChart";

export interface RunPreviewImage {
  run_dir: string;
  name: string;
}

const props = defineProps({
  scalars: { type: Object as PropType<Record<string, ScalarPoint[]>>, default: () => ({}) },
  previewImages: { type: Array as PropType<RunPreviewImage[]>, default: () => [] },
  loading: { type: Boolean, default: false },
  /** When false, use a lighter dim (auto-refresh) instead of a full overlay. */
  loadingStrong: { type: Boolean, default: true },
});

const stepJumpTag = computed(() => resolveStepJumpTag(props.scalars));
const stepJumpTitle = computed(() => stepJumpPanelTitle(stepJumpTag.value));
const stepJumpValueLabel = computed(() => {
  if (stepJumpTag.value === "train/automagic_avg_lr") return "lr";
  if (stepJumpTag.value === "train/prodigy_d") return "D";
  return "grad norm";
});
</script>

<style scoped>
.run-loss-monitor {
  position: relative;
  width: 100%;
}
.run-loss-monitor--loading::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 2;
  border-radius: 6px;
  pointer-events: none;
  background: var(--el-bg-color);
  opacity: 0.55;
  transition: opacity 0.2s ease;
}
.run-loss-monitor--loading-subtle::after {
  opacity: 0.28;
}
.monitor-col {
  margin-bottom: 16px;
}
.monitor-panel {
  min-height: 148px;
  padding: 8px 10px 10px;
  background: var(--el-fill-color-blank);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
}
.monitor-panel--images {
  min-height: 148px;
}
.panel-title {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
}
.panel-title--hint {
  cursor: help;
  border-bottom: 1px dotted var(--el-border-color);
  display: inline-block;
}
.panel-empty {
  display: block;
  padding: 24px 0;
  line-height: 1.45;
}
</style>

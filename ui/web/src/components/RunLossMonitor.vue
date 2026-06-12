<template>
  <div
    class="run-loss-monitor"
    :class="{
      'run-loss-monitor--loading': loading,
      'run-loss-monitor--loading-subtle': loading && !loadingStrong,
    }"
  >
    <!-- TensorBoard-style refresh: manual reload + opt-in auto-update at a chosen cadence. Off by
         default so the charts stay put unless you ask for live updates. -->
    <div class="loss-monitor__toolbar">
      <el-button size="small" :icon="Refresh" :loading="loading" @click="emit('refresh')">
        Reload
      </el-button>
      <el-checkbox v-model="autoUpdate" size="small">Auto-update</el-checkbox>
      <span class="loss-monitor__cadence" :class="{ 'loss-monitor__cadence--off': !autoUpdate }">
        every
        <el-input-number
          v-model="cadenceSec"
          :min="1"
          :max="3600"
          :step="1"
          :controls="false"
          :disabled="!autoUpdate"
          size="small"
          class="loss-monitor__cadence-input"
        />
        s
      </span>
    </div>

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
import { computed, onUnmounted, ref, watch } from "vue";
import type { PropType } from "vue";
import { Refresh } from "@element-plus/icons-vue";
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
  step?: number | null;
  prompt?: string;
}

const props = defineProps({
  scalars: { type: Object as PropType<Record<string, ScalarPoint[]>>, default: () => ({}) },
  previewImages: { type: Array as PropType<RunPreviewImage[]>, default: () => [] },
  loading: { type: Boolean, default: false },
  /** When false, use a lighter dim (auto-refresh) instead of a full overlay. */
  loadingStrong: { type: Boolean, default: true },
});

const emit = defineEmits<{ refresh: [] }>();

// Auto-update is opt-in (off by default), TensorBoard-style: when enabled we ask the parent to
// re-fetch metrics every `cadenceSec` seconds; the parent owns the actual fetch.
const autoUpdate = ref(false);
const cadenceSec = ref(10);
let timer: ReturnType<typeof setInterval> | null = null;

function clearTimer(): void {
  if (timer) clearInterval(timer);
  timer = null;
}

function restartTimer(): void {
  clearTimer();
  if (autoUpdate.value && cadenceSec.value > 0) {
    timer = setInterval(() => emit("refresh"), cadenceSec.value * 1000);
  }
}

// Enabling auto-update refreshes once immediately, then on the cadence; changing the cadence just
// reschedules without an extra fetch.
watch(autoUpdate, (on) => {
  if (on) emit("refresh");
  restartTimer();
});
watch(cadenceSec, restartTimer);
onUnmounted(clearTimer);

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
.loss-monitor__toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 12px;
}
.loss-monitor__cadence {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.loss-monitor__cadence--off {
  opacity: 0.6;
}
.loss-monitor__cadence-input {
  width: 64px;
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

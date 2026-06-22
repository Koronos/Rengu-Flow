<template>
  <div class="run-loss-monitor">
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
          <MetricOverlayChart
            metric="train/epoch_loss"
            :runs="runRef"
            :series="epochLossSeries"
            output-dir=""
            :smoothing="0"
            :log-scale="false"
            sync-key="detail-epoch-loss"
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
          <MetricOverlayChart
            metric="train/loss"
            :runs="runRef"
            :series="stepLossSeries"
            output-dir=""
            :smoothing="0"
            :log-scale="false"
            sync-key="detail-step-loss"
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
          <MetricOverlayChart
            :metric="stepJumpTag"
            :runs="runRef"
            :series="stepJumpSeries"
            output-dir=""
            :smoothing="0"
            :log-scale="false"
            sync-key="detail-step-jump"
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
import MetricOverlayChart from "./MetricOverlayChart.vue";
import { colorForRun } from "../lib/runColors";
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
  /** Display name + legend label for this run's single series in the charts. */
  runName: { type: String, default: "run" },
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

// Single-run adapters so the curated panels reuse the compare chart (MetricOverlayChart): one run
// ref + the pre-loaded series for each panel's tag, keyed by that run id.
const runRef = computed(() => [{ id: "run", name: props.runName, color: colorForRun(0) }]);
const epochLossSeries = computed(() => ({ run: props.scalars["train/epoch_loss"] || [] }));
const stepLossSeries = computed(() => ({ run: props.scalars["train/loss"] || [] }));
const stepJumpSeries = computed(() => ({ run: props.scalars[stepJumpTag.value] || [] }));
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

<template>
  <el-card v-if="run" shadow="never" class="train-live-panel">
    <template #header>
      <div class="live-head">
        <span class="live-title">
          <span class="pulse-dot" />
          Training live
        </span>
        <el-space wrap class="live-head-actions">
          <el-tag
            v-if="streamStatus"
            size="small"
            :type="streamTagType"
            class="stream-status-tag"
          >
            {{ streamStatusLabel }}
          </el-tag>
          <slot name="header-extra" />
          <el-button size="small" type="primary" @click="$emit('open-detail', run)">
            Open detail
          </el-button>
        </el-space>
      </div>
    </template>

    <!-- The body is a compact, clickable summary; full controls (signals, previews,
         charts, log) live on the run detail page. -->
    <div
      class="live-clickable"
      role="button"
      tabindex="0"
      @click="$emit('open-detail', run)"
      @keydown.enter="$emit('open-detail', run)"
    >
      <div class="live-meta">
        <code v-if="run.label || run.run_name">{{ run.label || run.run_name }}</code>
        <el-text type="info" size="small">{{ run.num_gpus }} GPU</el-text>
      </div>

      <el-alert
        v-if="diskExportWait"
        type="warning"
        show-icon
        :closable="false"
        title="Paused — free disk space, then continue export (open detail)"
        class="live-disk-alert"
      />

      <div v-if="caching" class="live-progress">
        <div class="progress-labels">
          <span>
            Caching {{ progress?.current ?? 0 }}
            <template v-if="progress?.total"> / {{ progress?.total }}</template>
          </span>
        </div>
        <el-progress
          v-if="progress?.percent != null"
          :percentage="progress.percent"
          :stroke-width="12"
          :show-text="true"
        />
        <el-progress v-else :percentage="0" :indeterminate="true" :stroke-width="12" />
      </div>

      <div v-else-if="progress" class="live-progress">
        <el-progress
          v-if="progress.percent != null"
          :percentage="Math.min(100, Math.round(progress.percent))"
          :stroke-width="14"
          :show-text="true"
          class="live-progress-bar"
        />
        <div class="progress-readout">
          <span v-if="progress.step != null" class="live-step">
            step {{ progress.step }}<template v-if="progress.max_steps"> / {{ progress.max_steps }}</template>
          </span>
          <span v-else class="live-step">Waiting for first step…</span>
          <template v-if="epochInfo">
            <span class="live-sep">·</span>
            <span class="live-epoch">
              epoch {{ epochInfo.cur }}<template v-if="epochInfo.total != null"> / {{ epochInfo.total }}</template><template v-if="epochInfo.left != null"> ({{ epochInfo.left }} left)</template>
            </span>
          </template>
          <span v-if="displayLoss != null" class="live-sep">·</span>
          <span v-if="displayLoss != null" class="live-loss" :title="lossTitle">
            loss {{ formatLoss(displayLoss) }}
          </span>
          <span v-if="valLoss != null" class="live-sep">·</span>
          <span v-if="valLoss != null" class="live-val" title="held-out validation loss">
            val {{ formatLoss(valLoss) }}
          </span>
          <span v-if="valGap != null" class="live-sep">·</span>
          <span
            v-if="valGap != null"
            class="live-gap"
            :class="{ 'live-gap-warn': valGap > 0 }"
            title="train-val gap (val − train probe); rising = overfitting"
          >
            gap {{ formatLoss(valGap) }}
          </span>
          <span v-if="progressHint" class="live-sep">·</span>
          <span v-if="progressHint" class="live-speed">{{ progressHint }}</span>
        </div>
      </div>

      <el-text type="info" size="small" class="live-open-hint">
        Open detail for signals, previews & charts →
      </el-text>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { LiveStreamStatus } from "../composables/useTrainLiveStream";
import type { PropType } from "vue";
import { formatRunProgressHint } from "../lib/formatRunProgress";
import type { TrainingRunRow } from "../types/api";

interface PreviewImage {
  run_dir: string;
  name: string;
}

type TrainLiveRun = TrainingRunRow & {
  scalars?: Record<string, { step: number; value: number }[]>;
  preview_images?: PreviewImage[];
};

const props = defineProps({
  run: { type: Object as PropType<TrainLiveRun | null>, default: null },
  streamStatus: { type: String as PropType<LiveStreamStatus | "">, default: "" },
  // Accepted for compatibility with the runs page bindings; the heavy views (charts,
  // log, signals) now live on the run detail page, so these are intentionally unused here.
  metricsLoading: { type: Boolean, default: false },
  logText: { type: String, default: "" },
  streamError: { type: String, default: "" },
  showTrainingLog: { type: Boolean, default: true },
});

defineEmits(["open-detail", "stop", "signal"]);

const progress = computed(() => props.run?.progress || null);
const progressHint = computed(() => formatRunProgressHint(progress.value));

const epochInfo = computed(() => {
  const p = progress.value;
  if (!p || p.epoch == null) return null;
  const total = p.epochs ?? null;
  const left = total != null ? Math.max(0, total - p.epoch) : null;
  return { cur: p.epoch, total, left };
});

// Show the Kohya-style moving-average loss (steady) when available; fall back to the
// instant per-step loss. The tooltip surfaces the raw value when smoothing is shown.
const displayLoss = computed(() => {
  const p = progress.value;
  if (!p) return null;
  return p.loss_avg ?? p.loss ?? null;
});
const lossTitle = computed(() => {
  const p = progress.value;
  if (!p || p.loss_avg == null || p.loss == null) return "";
  return `avg loss (last steps); instant ${p.loss.toFixed(6)}`;
});

// Generalization probe: held-out validation loss and the train-val gap (overfitting signal).
const valLoss = computed(() => progress.value?.val_loss ?? null);
const valGap = computed(() => progress.value?.val_gap ?? null);

const streamStatusLabel = computed(() => {
  switch (props.streamStatus) {
    case "connected":
      return "Live";
    case "reconnecting":
      return "Reconnecting…";
    case "offline":
      return "Polling";
    default:
      return "";
  }
});

const streamTagType = computed((): "success" | "warning" | "info" => {
  if (props.streamStatus === "connected") return "success";
  if (props.streamStatus === "reconnecting") return "warning";
  return "info";
});

const diskExportWait = computed(() => progress.value?.phase === "waiting_disk_export");
const caching = computed(() => progress.value?.phase === "caching");

function formatLoss(v: number | null | undefined): string | number | null | undefined {
  return typeof v === "number" ? v.toFixed(6) : v;
}
</script>

<style scoped>
.train-live-panel {
  margin-bottom: 16px;
  border-color: var(--el-color-success-light-5);
}
.live-disk-alert {
  margin-bottom: 12px;
}
.live-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.live-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}
.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--el-color-success);
  animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}
.live-clickable {
  cursor: pointer;
  border-radius: var(--el-border-radius-base);
  transition: background 0.15s ease;
}
.live-clickable:hover {
  background: var(--el-fill-color-light);
}
.live-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}
.live-progress {
  margin-bottom: 8px;
}
.progress-labels {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 6px;
  font-size: 13px;
}
.live-progress-bar {
  margin-bottom: 6px;
}
.progress-readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px;
  font-size: 13px;
  font-family: ui-monospace, monospace;
}
.live-step {
  font-weight: 600;
}
.live-sep {
  color: var(--el-text-color-secondary);
}
.live-epoch,
.live-loss,
.live-val,
.live-gap,
.live-speed {
  font-family: ui-monospace, monospace;
}
.live-speed {
  color: var(--el-text-color-secondary);
}
.live-val {
  color: var(--el-color-info);
}
.live-gap {
  color: var(--el-text-color-secondary);
}
.live-gap-warn {
  color: var(--el-color-warning);
}
.live-open-hint {
  display: block;
  margin-top: 6px;
}
.stream-status-tag {
  font-variant-numeric: tabular-nums;
}
</style>

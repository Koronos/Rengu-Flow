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
          <el-button size="small" @click="$emit('open-detail', run)">Open detail</el-button>
          <el-button
            size="small"
            type="primary"
            :disabled="diskExportWait"
            @click="onSignal('save_quit')"
          >
            Stop &amp; checkpoint
          </el-button>
          <el-tooltip content="Force-kill (no checkpoint)" :show-after="300">
            <el-button size="small" type="danger" plain @click="$emit('stop', run.job_id)">
              Force stop
            </el-button>
          </el-tooltip>
        </el-space>
      </div>
    </template>

    <div class="live-meta">
      <code v-if="run.label || run.run_name">{{ run.label || run.run_name }}</code>
      <el-text type="info" size="small">
        {{ run.num_gpus }} GPU
      </el-text>
    </div>

    <el-alert
      v-if="diskExportWait"
      type="warning"
      show-icon
      :closable="false"
      title="Paused — free disk space, then continue export"
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
        :percentage="progress.percent"
        :stroke-width="14"
        :show-text="true"
        class="live-progress-bar"
      />
      <div class="progress-readout">
        <span v-if="progress.step != null" class="live-step">
          step {{ progress.step }}<template v-if="progress.max_steps"> / {{ progress.max_steps }}</template>
        </span>
        <span v-else class="live-step">Waiting for first step…</span>
        <span v-if="progress.loss != null" class="live-sep">·</span>
        <span v-if="progress.loss != null" class="live-loss">
          loss {{ formatLoss(progress.loss) }}
        </span>
        <span v-if="progressHint" class="live-sep">·</span>
        <span v-if="progressHint" class="live-speed">{{ progressHint }}</span>
      </div>
    </div>

    <RunSignalActions
      :available="signalsAvailable"
      :disk-export-wait="diskExportWait"
      :show-unavailable-hint="false"
      compact
      @send="onSignal"
    />

    <RunLossMonitor
      class="live-charts"
      :scalars="run.scalars || {}"
      :preview-images="run.preview_images || []"
      :loading="metricsLoading"
      :loading-strong="false"
    />

    <el-collapse v-if="showTrainingLog" v-model="logCollapse" class="live-log-collapse">
      <el-collapse-item name="log">
        <template #title>
          <span class="live-log-title">Training log</span>
          <el-text v-if="streamError" type="warning" size="small" class="live-log-warn">
            {{ streamError }}
          </el-text>
        </template>
        <pre ref="logPreRef" class="live-log-pre" @scroll="onLogScroll">{{ logText || "(waiting for output…)" }}</pre>
      </el-collapse-item>
    </el-collapse>
  </el-card>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { LiveStreamStatus } from "../composables/useTrainLiveStream";
import RunLossMonitor from "./RunLossMonitor.vue";
import RunSignalActions from "./RunSignalActions.vue";
import { jobSignalsAvailable } from "../lib/trainingSignals";
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
  metricsLoading: { type: Boolean, default: false },
  logText: { type: String, default: "" },
  streamStatus: { type: String as PropType<LiveStreamStatus | "">, default: "" },
  streamError: { type: String, default: "" },
  showTrainingLog: { type: Boolean, default: true },
});

const emit = defineEmits(["open-detail", "stop", "signal"]);

const logCollapse = ref<string[]>(["log"]);
const logPreRef = ref<HTMLElement | null>(null);
let userScrolledUp = false;

const progress = computed(() => props.run?.progress || null);
const progressHint = computed(() => formatRunProgressHint(progress.value));

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

function onLogScroll(): void {
  const el = logPreRef.value;
  if (!el) return;
  const threshold = 48;
  userScrolledUp = el.scrollTop + el.clientHeight < el.scrollHeight - threshold;
}

async function scrollLogToEnd(): Promise<void> {
  await nextTick();
  const el = logPreRef.value;
  if (!el || userScrolledUp) return;
  el.scrollTop = el.scrollHeight;
}

watch(
  () => props.logText,
  () => {
    void scrollLogToEnd();
  }
);
const diskExportWait = computed(
  () => progress.value?.phase === "waiting_disk_export"
);
const caching = computed(() => progress.value?.phase === "caching");
const signalsAvailable = computed(() =>
  jobSignalsAvailable(props.run ? { state: props.run.state } : null)
);

function onSignal(type: string) {
  if (!props.run) return;
  emit("signal", props.run, type);
}

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
.live-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}
.live-progress {
  margin-bottom: 16px;
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
.live-loss,
.live-speed {
  font-family: ui-monospace, monospace;
}
.live-speed {
  color: var(--el-text-color-secondary);
}
.live-charts {
  margin-top: 4px;
}
.stream-status-tag {
  font-variant-numeric: tabular-nums;
}
.live-log-collapse {
  margin-top: 12px;
  border-top: 1px solid var(--el-border-color-lighter);
  padding-top: 4px;
}
.live-log-title {
  font-weight: 500;
  margin-right: 8px;
}
.live-log-warn {
  margin-left: 8px;
}
.live-log-pre {
  margin: 0;
  max-height: 280px;
  overflow: auto;
  padding: 10px 12px;
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  font-size: 12px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--el-fill-color-darker);
  color: var(--el-text-color-primary);
  border-radius: var(--el-border-radius-base);
}
</style>

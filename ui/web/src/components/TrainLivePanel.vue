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

      <RunProgress :progress="progress" class="live-progress" />

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
import RunProgress from "./RunProgress.vue";
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
.live-open-hint {
  display: block;
  margin-top: 6px;
}
.stream-status-tag {
  font-variant-numeric: tabular-nums;
}
</style>

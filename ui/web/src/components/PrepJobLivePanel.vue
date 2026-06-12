<template>
  <el-card shadow="never" class="prep-live-panel">
    <template #header>
      <div class="prep-live-head">
        <span class="prep-live-title">
          <span class="pulse-dot" />
          Prep job live
        </span>
        <el-space wrap class="prep-live-head-actions">
          <el-tag
            size="small"
            :type="streamTagType"
            class="stream-status-tag"
          >
            {{ streamStatusLabel }}
          </el-tag>
          <el-tag size="small" type="info" effect="plain">{{ stageLabel }}</el-tag>
          <el-button
            size="small"
            type="danger"
            plain
            :loading="stopping"
            @click="onStop"
          >
            Stop
          </el-button>
        </el-space>
      </div>
    </template>

    <!-- Progress -->
    <div class="prep-live-body">
      <div v-if="progress" class="prep-live-progress">
        <el-progress
          v-if="progress.percent != null"
          :percentage="Math.round(progress.percent * 100)"
          :status="progressStatus"
          striped
          striped-flow
          :duration="6"
        />
        <div class="prep-live-meta">
          <el-text size="small" type="info" class="prep-live-phase">
            {{ progress.phase ?? stageLabel }}
          </el-text>
          <el-text v-if="progress.step != null" size="small" type="info">
            {{ progress.step }}<template v-if="progress.max_steps != null">/{{ progress.max_steps }}</template>
          </el-text>
          <el-text v-if="progress.msg" size="small" class="prep-live-msg">
            {{ progress.msg }}
          </el-text>
        </div>
      </div>
      <el-text v-else size="small" type="info">Waiting for progress…</el-text>

      <!-- Log tail -->
      <div class="prep-live-log">
        <pre class="prep-log-pre">{{ logText || "(waiting for output…)" }}</pre>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import type { PropType } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { usePrepJobLive } from "../composables/usePrepJobLive";
import { useJobLogStream } from "../composables/useJobLogStream";
import { formatError } from "../lib/formatError";
import type { PrepStage } from "../types/api";

const props = defineProps({
  jobId: { type: String, required: true },
  stage: { type: String as PropType<PrepStage>, required: true },
});

const emit = defineEmits<{
  (e: "stopped"): void;
}>();

const stopping = ref(false);

const { progress, streamStatus, streamError } = usePrepJobLive(
  () => props.jobId,
  { onRunFinished: () => emit("stopped") }
);

const { logText } = useJobLogStream(() => props.jobId);

const stageLabel = computed(() => {
  const map: Record<string, string> = { tag: "Tag", caption: "Caption", clean: "Clean" };
  return map[props.stage] ?? props.stage;
});

const streamStatusLabel = computed(() => {
  switch (streamStatus.value) {
    case "connected": return "Live";
    case "reconnecting": return "Reconnecting…";
    default: return "Offline";
  }
});

const streamTagType = computed((): "success" | "warning" | "info" => {
  if (streamStatus.value === "connected") return "success";
  if (streamStatus.value === "reconnecting") return "warning";
  return "info";
});

const progressStatus = computed(() => {
  if (!progress.value) return undefined;
  const pct = progress.value.percent ?? 0;
  return pct >= 1 ? "success" : undefined;
});

async function onStop(): Promise<void> {
  stopping.value = true;
  try {
    await api.stopJob(props.jobId);
    ElMessage.info("Stop requested");
    emit("stopped");
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    stopping.value = false;
  }
}

// Expose streamError for parent if needed
defineExpose({ streamError });
</script>

<style scoped>
.prep-live-panel {
  border-color: var(--el-color-success-light-5);
}
.prep-live-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.prep-live-title {
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
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
.stream-status-tag {
  font-variant-numeric: tabular-nums;
}
.prep-live-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.prep-live-progress {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.prep-live-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.prep-live-phase {
  font-family: var(--rf-font-mono);
  font-size: 11px;
}
.prep-live-msg {
  color: var(--el-text-color-primary);
  font-size: 12px;
}
.prep-live-log {
  background: var(--el-fill-color-darker, #1a1a1a);
  border-radius: var(--el-border-radius-base);
  padding: 8px 12px;
  max-height: 220px;
  overflow-y: auto;
}
.prep-log-pre {
  font-family: var(--rf-font-mono);
  font-size: 11px;
  line-height: 1.5;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--el-text-color-primary);
}
</style>

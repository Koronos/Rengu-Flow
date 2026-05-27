<template>
  <el-card v-if="run" shadow="never" class="train-live-panel">
    <template #header>
      <div class="live-head">
        <span class="live-title">
          <span class="pulse-dot" />
          Training live
        </span>
        <el-space wrap>
          <el-button size="small" @click="$emit('open-detail', run)">Open detail</el-button>
          <el-button size="small" type="danger" plain @click="$emit('stop', run.job_id)">
            Stop
          </el-button>
        </el-space>
      </div>
    </template>

    <div class="live-meta">
      <code v-if="run.label || run.run_name">{{ run.label || run.run_name }}</code>
      <el-text type="info" size="small">
        config {{ run.config_id ?? "—" }} · {{ run.num_gpus }} GPU
      </el-text>
    </div>

    <div v-if="progress" class="live-progress">
      <div class="progress-labels">
        <span v-if="progress.step != null">
          Step {{ progress.step }}
          <template v-if="progress.max_steps"> / {{ progress.max_steps }}</template>
        </span>
        <span v-else>Waiting for first step…</span>
        <span v-if="progress.loss != null" class="live-loss">
          loss {{ formatLoss(progress.loss) }}
        </span>
      </div>
      <el-progress
        v-if="progress.percent != null"
        :percentage="progress.percent"
        :stroke-width="10"
        :show-text="true"
      />
      <el-text v-else-if="!progress.status_available" type="warning" size="small">
        Enable <code>monitoring.enable_status_file</code> in config for faster progress updates.
      </el-text>
    </div>

    <el-row :gutter="16" class="live-charts">
      <el-col :xs="24" :md="14">
        <ScalarLineChart :scalars="run.scalars || {}" tag="train/loss" />
      </el-col>
      <el-col :xs="24" :md="10">
        <div v-if="run.preview_images?.length" class="preview-strip">
          <el-text type="info" size="small" class="preview-label">Previews</el-text>
          <div class="preview-grid">
            <a
              v-for="img in run.preview_images.slice(0, 6)"
              :key="img.name"
              :href="previewUrl(img)"
              target="_blank"
              rel="noopener"
              class="preview-thumb"
            >
              <img :src="previewUrl(img)" :alt="img.name" loading="lazy" />
            </a>
          </div>
        </div>
        <el-text v-else type="info" size="small">No preview PNGs yet</el-text>
      </el-col>
    </el-row>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import ScalarLineChart from "./ScalarLineChart.vue";

const props = defineProps({
  run: { type: Object, default: null },
});

defineEmits(["open-detail", "stop"]);

const progress = computed(() => props.run?.progress || null);

function formatLoss(v) {
  return typeof v === "number" ? v.toFixed(6) : v;
}

function previewUrl(img) {
  const params = new URLSearchParams({
    run_dir: img.run_dir,
    name: img.name,
  });
  return `/api/v1/train/preview-image?${params.toString()}`;
}
</script>

<style scoped>
.train-live-panel {
  margin-bottom: 16px;
  border-color: var(--el-color-success-light-5);
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
.live-loss {
  font-family: ui-monospace, monospace;
}
.live-charts {
  margin-top: 4px;
}
.preview-strip {
  width: 100%;
}
.preview-label {
  display: block;
  margin-bottom: 6px;
}
.preview-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}
.preview-thumb {
  display: block;
  aspect-ratio: 1;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid var(--el-border-color-lighter);
}
.preview-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
</style>

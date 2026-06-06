<template>
  <div v-if="progress" class="run-progress">
    <template v-if="caching">
      <div class="run-progress__readout">
        Caching<template v-if="progress.total"> {{ progress.current ?? 0 }} / {{ progress.total }}</template><template v-else>…</template>
      </div>
      <el-progress
        v-if="progress.percent != null"
        :percentage="pct"
        :stroke-width="12"
        :show-text="true"
      />
      <el-progress v-else :percentage="0" :indeterminate="true" :stroke-width="12" />
    </template>

    <template v-else>
      <el-progress
        v-if="progress.percent != null"
        :percentage="pct"
        :stroke-width="14"
        :show-text="true"
        class="run-progress__bar"
      />
      <div class="run-progress__readout">
        <span v-if="progress.step != null" class="run-progress__step">
          step {{ progress.step }}<template v-if="progress.max_steps"> / {{ progress.max_steps }}</template>
        </span>
        <span v-else class="run-progress__step">Waiting for first step…</span>
        <template v-if="epochInfo">
          <span class="run-progress__sep">·</span>
          <span>
            epoch {{ epochInfo.cur }}<template v-if="epochInfo.total != null"> / {{ epochInfo.total }}</template><template v-if="epochInfo.left != null"> ({{ epochInfo.left }} left)</template>
          </span>
        </template>
        <template v-if="displayLoss != null">
          <span class="run-progress__sep">·</span>
          <span :title="lossTitle">loss {{ formatLoss(displayLoss) }}</span>
        </template>
        <template v-if="valLoss != null">
          <span class="run-progress__sep">·</span>
          <span title="held-out validation loss">val {{ formatLoss(valLoss) }}</span>
        </template>
        <template v-if="valGap != null">
          <span class="run-progress__sep">·</span>
          <span
            :class="{ 'run-progress__gap-warn': valGap > 0 }"
            title="train-val gap (val − train probe); rising = overfitting"
          >gap {{ formatLoss(valGap) }}</span>
        </template>
        <template v-if="progressHint">
          <span class="run-progress__sep">·</span>
          <span class="run-progress__muted">{{ progressHint }}</span>
        </template>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, type PropType } from "vue";
import { formatRunProgressHint } from "../lib/formatRunProgress";
import type { RunProgress } from "../types/api";

const props = defineProps({
  progress: { type: Object as PropType<RunProgress | null>, default: null },
});

const caching = computed(() => props.progress?.phase === "caching");

const pct = computed(() => {
  const p = props.progress;
  if (!p || p.percent == null) return 0;
  return Math.min(100, Math.round(p.percent));
});

const epochInfo = computed(() => {
  const p = props.progress;
  if (!p || p.epoch == null) return null;
  const total = p.epochs ?? null;
  const left = total != null ? Math.max(0, total - p.epoch) : null;
  return { cur: p.epoch, total, left };
});

// Prefer the Kohya-style moving-average loss (steady); fall back to the instant loss.
const displayLoss = computed(() => {
  const p = props.progress;
  if (!p) return null;
  return p.loss_avg ?? p.loss ?? null;
});
const lossTitle = computed(() => {
  const p = props.progress;
  if (!p || p.loss_avg == null || p.loss == null) return "";
  return `avg loss (last steps); instant ${p.loss.toFixed(6)}`;
});

// Generalization probe: held-out validation loss and the train-val gap (overfitting signal).
const valLoss = computed(() => props.progress?.val_loss ?? null);
const valGap = computed(() => props.progress?.val_gap ?? null);

const progressHint = computed(() => formatRunProgressHint(props.progress));

function formatLoss(v: number | null | undefined): string | number | null | undefined {
  return typeof v === "number" ? v.toFixed(6) : v;
}
</script>

<style scoped>
.run-progress {
  width: 100%;
}
.run-progress__bar {
  margin-bottom: 6px;
}
.run-progress__readout {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px;
  font-size: 13px;
  font-family: ui-monospace, monospace;
}
.run-progress__step {
  font-weight: 600;
}
.run-progress__sep,
.run-progress__muted {
  color: var(--el-text-color-secondary);
}
.run-progress__gap-warn {
  color: var(--el-color-warning);
}
</style>

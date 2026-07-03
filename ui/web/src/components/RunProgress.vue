<template>
  <div v-if="progress" class="run-progress">
    <template v-if="caching">
      <div class="run-progress__readout">
        <template v-if="progress.stage">Caching {{ progress.stage }}/{{ progress.stages }}: {{ progress.stage_name }}<template v-if="progress.detail"> · {{ progress.detail }}</template><template v-if="progress.total"> ({{ progress.current ?? 0 }}/{{ progress.total }})</template></template>
        <template v-else>Caching<template v-if="progress.total"> {{ progress.current ?? 0 }} / {{ progress.total }}</template><template v-else>…</template></template>
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
      <div v-if="progress.percent != null" class="run-progress__track">
        <el-progress
          :percentage="pct"
          :stroke-width="14"
          :show-text="false"
          class="run-progress__bar"
        />
        <div class="run-progress__marks">
          <span
            v-for="m in epochMarks"
            :key="'e' + m.left"
            class="run-progress__epoch"
            :style="{ left: m.left }"
            :title="m.title"
          />
          <span
            v-for="m in eventMarks"
            :key="'v' + m.left + m.cls"
            class="run-progress__event"
            :class="m.cls"
            :style="{ left: m.left }"
            :title="m.title"
          />
        </div>
      </div>
      <div class="run-progress__readout">
        <span v-if="progress.step != null" class="run-progress__step">
          step {{ progress.step }}<template v-if="progress.max_steps"> / {{ progress.max_steps }}</template><span v-if="progress.percent != null" class="run-progress__pct"> ({{ pct }}%)</span>
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
  // Steps at which previews were generated / checkpoints were saved; rendered as ticks on the bar.
  previewSteps: { type: Array as PropType<number[]>, default: () => [] },
  checkpointSteps: { type: Array as PropType<number[]>, default: () => [] },
});

const caching = computed(() => props.progress?.phase === "caching");

// Epoch boundaries are evenly spaced in steps, so position i is just i/epochs (no max_steps needed).
const epochMarks = computed(() => {
  const n = props.progress?.epochs ?? 0;
  if (!n || n < 2) return [];
  return Array.from({ length: n - 1 }, (_, i) => ({
    left: `${((i + 1) / n) * 100}%`,
    title: `epoch ${i + 1}`,
  }));
});

// Preview + checkpoint ticks, merged when they land on the same step ("done together").
const eventMarks = computed(() => {
  const max = props.progress?.max_steps ?? 0;
  if (!max) return [];
  const byStep = new Map<number, { preview: boolean; checkpoint: boolean }>();
  const add = (s: number, k: "preview" | "checkpoint") => {
    if (s == null || s < 0) return;
    const e = byStep.get(s) ?? { preview: false, checkpoint: false };
    e[k] = true;
    byStep.set(s, e);
  };
  for (const s of props.previewSteps) add(s, "preview");
  for (const s of props.checkpointSteps) add(s, "checkpoint");
  return [...byStep.entries()].map(([step, e]) => {
    const both = e.preview && e.checkpoint;
    const kind = both ? "checkpoint + preview" : e.checkpoint ? "checkpoint" : "preview";
    return {
      left: `${Math.min(100, (step / max) * 100)}%`,
      cls: both ? "is-both" : e.checkpoint ? "is-checkpoint" : "is-preview",
      title: `${kind} · step ${step}`,
    };
  });
});

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
.run-progress__track {
  position: relative;
  margin-bottom: 6px;
}
.run-progress__bar {
  margin-bottom: 0;
}
.run-progress__marks {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.run-progress__epoch,
.run-progress__event {
  position: absolute;
  top: 0;
  bottom: 0;
  transform: translateX(-50%);
}
.run-progress__epoch {
  width: 1px;
  background: var(--el-text-color-secondary);
  opacity: 0.35;
}
.run-progress__event {
  width: 3px;
  border-radius: 2px;
  pointer-events: auto;
}
.run-progress__event.is-preview {
  background: var(--el-color-primary);
}
.run-progress__event.is-checkpoint {
  background: var(--el-color-success);
}
.run-progress__event.is-both {
  background: linear-gradient(90deg, var(--el-color-success) 50%, var(--el-color-primary) 50%);
}
.run-progress__pct {
  color: var(--el-text-color-secondary);
  font-weight: 400;
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

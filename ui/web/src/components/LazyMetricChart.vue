<template>
  <div ref="root" class="lazy-metric">
    <div class="metric-head">{{ tag }}</div>
    <div v-if="error" class="metric-state">
      <el-text type="danger" size="small">{{ error }}</el-text>
    </div>
    <div v-else-if="loading" class="metric-state">
      <el-text type="info" size="small">Loading…</el-text>
    </div>
    <div v-else-if="loaded" class="curve-row">
      <div v-for="r in runs" :key="r.id" class="curve-cell">
        <div class="curve-name">{{ r.name }}</div>
        <ScalarLineChart
          :scalars="scalarsFor(r.id)"
          :tag="tag"
          :width="280"
          :height="90"
          :max-points="600"
        />
      </div>
    </div>
    <div v-else class="metric-state placeholder">
      <el-text type="info" size="small">Scroll to load</el-text>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "../api";
import ScalarLineChart from "./ScalarLineChart.vue";
import type { ScalarMetricPoint } from "../types/api";

const props = defineProps<{
  tag: string;
  runs: { id: string; name: string }[];
  outputDir: string;
}>();

const root = ref<HTMLElement | null>(null);
const loading = ref(false);
const loaded = ref(false);
const error = ref("");
const series = ref<Record<string, ScalarMetricPoint[]>>({});
let observer: IntersectionObserver | null = null;

function scalarsFor(runId: string): Record<string, ScalarMetricPoint[]> {
  const points = series.value[runId];
  return points ? { [props.tag]: points } : {};
}

async function loadSeries(): Promise<void> {
  if (loaded.value || loading.value) return;
  loading.value = true;
  try {
    const ids = props.runs.map((r) => r.id);
    const res = await api.runSeries(ids, props.tag, 600, props.outputDir);
    series.value = res.series || {};
    loaded.value = true;
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  if (!root.value) return;
  // Lazy: only fetch this metric's series once its chart scrolls into view (with a margin so it
  // is ready just before). Fall back to eager load where IntersectionObserver is unavailable.
  if (typeof IntersectionObserver === "undefined") {
    void loadSeries();
    return;
  }
  observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        void loadSeries();
        observer?.disconnect();
        observer = null;
      }
    },
    { rootMargin: "200px" }
  );
  observer.observe(root.value);
});

onBeforeUnmount(() => {
  observer?.disconnect();
  observer = null;
});
</script>

<style scoped>
.lazy-metric {
  min-height: 120px;
}
.metric-head {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 8px;
}
.metric-state {
  display: flex;
  align-items: center;
  min-height: 90px;
}
.metric-state.placeholder {
  border: 1px dashed var(--el-border-color-lighter);
  border-radius: 4px;
  justify-content: center;
}
.curve-row {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}
.curve-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.curve-name {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>

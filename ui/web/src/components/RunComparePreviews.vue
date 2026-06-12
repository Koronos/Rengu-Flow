<template>
  <div ref="root" class="compare-previews">
    <div v-for="r in runs" :key="r.id" class="compare-previews__run">
      <div class="compare-previews__head">
        <span class="compare-previews__swatch" :style="{ background: r.color }"></span>
        <span class="compare-previews__name">{{ r.name }}</span>
      </div>
      <PreviewStepBrowser :preview-images="previewsByRun[r.id] || []" />
    </div>
    <div v-if="loading" class="compare-previews__hint">loading previews…</div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import PreviewStepBrowser from "./PreviewStepBrowser.vue";
import { api } from "../api";
import type { RunPreviewImageRef } from "../types/api";

interface RunRef {
  id: string;
  name: string;
  color: string;
}

const props = defineProps<{
  runs: RunRef[];
  outputDir: string;
}>();

const previewsByRun = ref<Record<string, RunPreviewImageRef[]>>({});
const loading = ref(false);
const root = ref<HTMLElement | null>(null);
let controller: AbortController | null = null;
let observer: IntersectionObserver | null = null;
let started = false;

async function load() {
  controller?.abort();
  if (!props.runs.length) {
    previewsByRun.value = {};
    return;
  }
  controller = new AbortController();
  const signal = controller.signal;
  loading.value = true;
  try {
    const entries = await Promise.all(
      props.runs.map(async (r) => {
        // run_id is the run folder name the previews endpoint resolves by.
        const res = await api.runPreviews(r.id, props.outputDir, signal);
        return [r.id, res.previews || []] as const;
      })
    );
    if (signal.aborted) return;
    previewsByRun.value = Object.fromEntries(entries);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") return;
    // Per-run failures are non-fatal — just leave that run without previews.
  } finally {
    if (!signal.aborted) loading.value = false;
  }
}

onMounted(() => {
  if (!root.value) return;
  const begin = () => {
    if (started) return;
    started = true;
    load();
  };
  if (typeof IntersectionObserver === "undefined") {
    begin();
  } else {
    observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          begin();
          observer?.disconnect();
          observer = null;
        }
      },
      { rootMargin: "200px" }
    );
    observer.observe(root.value);
  }
});

watch(
  () => props.runs.map((r) => r.id).join(","),
  () => {
    if (started) load();
  }
);

onBeforeUnmount(() => {
  controller?.abort();
  observer?.disconnect();
});
</script>

<style scoped>
.compare-previews {
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.compare-previews__run {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.compare-previews__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.compare-previews__swatch {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  flex: 0 0 auto;
}
.compare-previews__name {
  font-weight: 600;
  font-size: 13px;
}
.compare-previews__hint {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>

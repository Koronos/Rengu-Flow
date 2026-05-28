<template>
  <div
    ref="rootRef"
    class="preview-image"
    :class="{
      'preview-image--revealed': revealed,
      'preview-image--error': errored,
    }"
  >
    <div
      class="preview-image__shimmer"
      :class="{ 'preview-image__shimmer--hidden': revealed || errored }"
      aria-hidden="true"
    />
    <el-image
      :src="src"
      :fit="fit"
      :lazy="lazy"
      :preview-src-list="previewSrcList"
      :initial-index="initialIndex"
      :preview-teleported="previewTeleported"
      :z-index="previewZIndex"
      class="preview-image__el"
      @load="onLoad"
      @error="onError"
    >
      <template v-if="$slots.error" #error>
        <slot name="error" />
      </template>
    </el-image>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import type { PropType } from "vue";

const props = defineProps({
  src: { type: String, required: true },
  fit: { type: String as PropType<"fill" | "contain" | "cover" | "none" | "scale-down">, default: "cover" },
  lazy: { type: Boolean, default: true },
  previewSrcList: { type: Array as PropType<string[]>, default: undefined },
  initialIndex: { type: Number, default: undefined },
  /** When preview-src-list is set, mount the lightbox on document.body (avoids dialog/transform traps). */
  previewTeleported: { type: Boolean, default: true },
  previewZIndex: { type: Number, default: 3000 },
});

const rootRef = ref<HTMLElement | null>(null);
const revealed = ref(false);
const errored = ref(false);

/** Minimum time the shimmer stays visible so fast/cached loads still feel soft. */
const MIN_SHIMMER_MS = 120;

let revealGeneration = 0;
let loadStartedAt = 0;

function resetState() {
  revealGeneration += 1;
  revealed.value = false;
  errored.value = false;
  loadStartedAt = performance.now();
}

function innerImg(): HTMLImageElement | null {
  return rootRef.value?.querySelector<HTMLImageElement>(".el-image__inner") ?? null;
}

function waitFrames(count = 2): Promise<void> {
  return new Promise((resolve) => {
    let left = count;
    const step = () => {
      left -= 1;
      if (left <= 0) resolve();
      else requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function waitMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function revealFromImg(img: HTMLImageElement | null, generation: number) {
  if (!img || generation !== revealGeneration || errored.value || revealed.value) return;

  try {
    if (typeof img.decode === "function") await img.decode();
  } catch {
    // Still show if decode fails but the browser has pixels.
  }

  if (generation !== revealGeneration || errored.value) return;

  const elapsed = performance.now() - loadStartedAt;
  if (elapsed < MIN_SHIMMER_MS) {
    await waitMs(MIN_SHIMMER_MS - elapsed);
  }
  if (generation !== revealGeneration || errored.value) return;

  await waitFrames(2);
  if (generation !== revealGeneration || errored.value || revealed.value) return;

  revealed.value = true;
}

function scheduleReveal() {
  const generation = revealGeneration;
  void nextTick().then(async () => {
    await revealFromImg(innerImg(), generation);
  });
}

function onLoad() {
  scheduleReveal();
}

function onError() {
  errored.value = true;
}

function probeCached() {
  const img = innerImg();
  if (img?.complete && img.naturalWidth > 0) scheduleReveal();
}

watch(
  () => props.src,
  () => {
    resetState();
    void nextTick(probeCached);
  }
);

onMounted(() => {
  loadStartedAt = performance.now();
  void nextTick(probeCached);
});
</script>

<style scoped>
.preview-image {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--el-fill-color-darker);
  content-visibility: auto;
}
.preview-image__shimmer {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
  opacity: 1;
  transition: opacity 0.28s ease-out;
  background: linear-gradient(
    90deg,
    var(--el-fill-color-darker) 0%,
    var(--el-fill-color-light) 45%,
    var(--el-fill-color-darker) 90%
  );
  background-size: 200% 100%;
  animation: preview-image-shimmer 1.35s ease-in-out infinite;
}
.preview-image__shimmer--hidden {
  opacity: 0;
  animation: none;
}
@keyframes preview-image-shimmer {
  0% {
    background-position: 100% 0;
  }
  100% {
    background-position: -100% 0;
  }
}
.preview-image__el {
  width: 100%;
  height: 100%;
  display: block;
}
.preview-image__el :deep(.el-image__inner) {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0;
  transition: opacity 0.32s ease-out;
}
.preview-image--revealed :deep(.el-image__inner) {
  opacity: 1;
}
</style>

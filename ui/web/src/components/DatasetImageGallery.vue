<template>
  <div class="image-gallery" :class="{ 'image-gallery--expanded': expanded }">
    <div v-if="loading" class="gallery-loading">
      <el-skeleton :rows="2" animated />
    </div>
    <el-alert
      v-else-if="error"
      type="warning"
      :title="error"
      show-icon
      :closable="false"
    />
    <el-empty
      v-else-if="!hasConfiguredPaths"
      description="Add a directory path to preview images"
      :image-size="40"
    />
    <el-empty
      v-else-if="!images.length && !loading"
      description="No images found in configured folders"
      :image-size="40"
    />
    <template v-else>
      <p v-if="summary" class="gallery-summary">{{ summary }}</p>
      <el-scrollbar :max-height="scrollbarMaxHeight">
        <div class="thumb-grid">
          <button
            v-for="(img, index) in images"
            :key="`${img.directory_index}-${img.name}`"
            type="button"
            class="thumb-cell"
            @click="openViewer(index)"
          >
            <PreviewImage
              :src="imageUrl(img.token)"
              class="thumb-img"
              :lazy="false"
            />
            <span class="thumb-label" :title="img.name">{{ img.name }}</span>
          </button>
        </div>
      </el-scrollbar>
      <el-button
        v-if="canLoadMore"
        class="load-more"
        size="small"
        :loading="loadingMore"
        @click="loadMore"
      >
        Load more
      </el-button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import PreviewImage from "./PreviewImage.vue";
import { useDatasetImageViewer } from "../composables/useDatasetImageViewer";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { formatMediaCount } from "../lib/formatMediaCount";
import type { DatasetPreviewImage } from "../types/api";

const props = defineProps({
  content: { type: String, default: "" },
  directoryIndex: { type: Number, default: null },
  /** Taller grid when opened in fullscreen gallery dialog. */
  expanded: { type: Boolean, default: false },
});

/** In dialog: leave room for summary + Load more below the scroll area. */
const scrollbarMaxHeight = computed(() =>
  props.expanded ? "min(62vh, calc(92vh - 14rem))" : "42vh"
);

const loading = ref(false);
const loadingMore = ref(false);
const error = ref("");
const images = ref<DatasetPreviewImage[]>([]);
const total = ref(0);
const totalCapped = ref(false);
const hasConfiguredPaths = ref(false);
const limit = 24;

const summary = computed(() => {
  if (!total.value) return "";
  const shown = images.value.length;
  const totalLabel = formatMediaCount(total.value, totalCapped.value);
  if (shown >= total.value && !totalCapped.value) {
    return `${shown} image${shown === 1 ? "" : "s"}`;
  }
  return `Showing ${shown} of ${totalLabel}`;
});

const canLoadMore = computed(() => images.value.length < total.value);

const previewList = computed(() =>
  images.value.map((img) => imageUrl(img.token))
);

const { openDatasetImageViewer, closeDatasetImageViewer } = useDatasetImageViewer();

function openViewer(index: number) {
  openDatasetImageViewer(previewList.value, index);
}

function imageUrl(token: string): string {
  return api.datasetPreviewImageUrl(token);
}

async function fetchImages({ append = false } = {}) {
  const content = props.content?.trim();
  if (!content) {
    images.value = [];
    total.value = 0;
    error.value = "";
    hasConfiguredPaths.value = false;
    return;
  }

  if (append) loadingMore.value = true;
  else loading.value = true;
  error.value = "";

  try {
    const offset = append ? images.value.length : 0;
    const body: {
      content: string;
      limit: number;
      offset: number;
      directory_index?: number;
    } = {
      content,
      limit,
      offset,
    };
    if (props.directoryIndex != null) {
      body.directory_index = props.directoryIndex;
    }
    const r = (await api.listDatasetPreviewImages(body)) as {
      ok?: boolean;
      error?: string;
      directories?: { path?: string; ok?: boolean; error?: string }[];
      total?: number;
      total_capped?: boolean;
      images?: DatasetPreviewImage[];
    };
    if (!r.ok) {
      error.value = r.error || "Could not load images";
      if (!append) images.value = [];
      return;
    }
    hasConfiguredPaths.value = (r.directories || []).some((d) => d.path);
    total.value = r.total ?? 0;
    totalCapped.value = !!r.total_capped;
    const batch = r.images || [];
    images.value = append ? [...images.value, ...batch] : batch;
    if (!batch.length && !hasConfiguredPaths.value) {
      error.value = "";
    } else if (!batch.length && hasConfiguredPaths.value) {
      const bad = (r.directories || []).filter((d) => !d.ok);
      if (bad.length && bad.every((d) => !d.ok)) {
        error.value = bad[0]?.error || "Configured paths are not accessible";
      }
    }
  } catch (e) {
    error.value = formatError(e);
    if (!append) images.value = [];
  } finally {
    loading.value = false;
    loadingMore.value = false;
  }
}

function loadMore() {
  if (!canLoadMore.value || loadingMore.value) return;
  fetchImages({ append: true });
}

watch(
  () => [props.content, props.directoryIndex],
  () => {
    closeDatasetImageViewer();
    fetchImages();
  },
  { immediate: true }
);
</script>

<style scoped>
.image-gallery {
  margin-top: 12px;
}
.image-gallery--expanded {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}
.image-gallery--expanded :deep(.el-scrollbar) {
  flex: 1;
  min-height: 0;
}
.gallery-summary {
  margin: 0 0 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.thumb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
  gap: 8px;
}
.image-gallery--expanded .thumb-grid {
  grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
  gap: 10px;
}
.thumb-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  padding: 0;
  border: none;
  background: transparent;
  text-align: left;
  cursor: zoom-in;
}
.thumb-cell:focus-visible {
  outline: 2px solid var(--el-color-primary);
  outline-offset: 2px;
  border-radius: 4px;
}
.thumb-img {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 4px;
  overflow: hidden;
  background: var(--el-fill-color-light);
}
.thumb-label {
  font-size: 10px;
  color: var(--el-text-color-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.load-more {
  margin-top: 8px;
  width: 100%;
  flex-shrink: 0;
}
.image-gallery--expanded .gallery-summary {
  flex-shrink: 0;
}
.gallery-loading {
  padding: 4px 0;
}
</style>

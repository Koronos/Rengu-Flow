<template>
  <div class="image-gallery">
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
      <el-scrollbar max-height="42vh">
        <div class="thumb-grid">
          <div
            v-for="img in images"
            :key="`${img.directory_index}-${img.name}`"
            class="thumb-cell"
          >
            <el-image
              :src="imageUrl(img.token)"
              :alt="img.name"
              fit="cover"
              lazy
              class="thumb-img"
              :preview-src-list="previewList"
              :initial-index="previewIndex(img)"
            />
            <span class="thumb-label" :title="img.name">{{ img.name }}</span>
          </div>
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

<script setup>
import { computed, ref, watch } from "vue";
import { api } from "../api";

const props = defineProps({
  content: { type: String, default: "" },
  directoryIndex: { type: Number, default: null },
});

const loading = ref(false);
const loadingMore = ref(false);
const error = ref("");
const images = ref([]);
const total = ref(0);
const hasConfiguredPaths = ref(false);
const limit = 24;

const summary = computed(() => {
  if (!total.value) return "";
  const shown = images.value.length;
  if (shown >= total.value) {
    return `${shown} image${shown === 1 ? "" : "s"}`;
  }
  return `Showing ${shown} of ${total.value}`;
});

const canLoadMore = computed(() => images.value.length < total.value);

const previewList = computed(() =>
  images.value.map((img) => imageUrl(img.token))
);

function imageUrl(token) {
  return api.datasetPreviewImageUrl(token);
}

function previewIndex(img) {
  return images.value.findIndex(
    (i) => i.directory_index === img.directory_index && i.name === img.name
  );
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
    const body = {
      content,
      limit,
      offset,
    };
    if (props.directoryIndex != null) {
      body.directory_index = props.directoryIndex;
    }
    const r = await api.listDatasetPreviewImages(body);
    if (!r.ok) {
      error.value = r.error || "Could not load images";
      if (!append) images.value = [];
      return;
    }
    hasConfiguredPaths.value = (r.directories || []).some((d) => d.path);
    total.value = r.total ?? 0;
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
    error.value = String(e);
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
    fetchImages();
  },
  { immediate: true }
);
</script>

<style scoped>
.image-gallery {
  margin-top: 12px;
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
.thumb-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.thumb-img {
  width: 100%;
  aspect-ratio: 1;
  border-radius: 4px;
  overflow: hidden;
  background: var(--el-fill-color-light);
}
.thumb-img :deep(.el-image__inner) {
  width: 100%;
  height: 100%;
  object-fit: cover;
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
}
.gallery-loading {
  padding: 4px 0;
}
</style>

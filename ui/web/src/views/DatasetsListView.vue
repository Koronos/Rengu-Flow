<template>
  <div class="datasets-page page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Compose folders into reusable dataset TOML</p>
      </div>
      <div class="page-head-actions">
        <el-button type="primary" :icon="Plus" @click="router.push({ name: 'datasets-new' })">
          New dataset
        </el-button>
      </div>
    </div>

    <div class="page-toolbar">
      <el-input
        v-model="query"
        clearable
        placeholder="Search by name or ID…"
        class="page-toolbar-search"
        :prefix-icon="Search"
        @input="scheduleSearch"
        @clear="load"
      />
      <LibrarySortControls
        v-model:sort-field="sortField"
        :sort-order="sortOrder"
        :field-options="fieldOptions"
        :order-button-label="orderButtonLabel"
        @toggle-order="onToggleSortOrder"
      />
      <DatasetViewModeToggle v-model="viewMode" />
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <div v-loading="loading" class="list-body">
      <el-empty v-if="!loading && !previewItems.length" description="No datasets yet" :image-size="64" />
      <DatasetPreviewCollection
        v-else
        :items="previewItems"
        :view-mode="viewMode"
        table-subtitle-label="Summary"
        @item-click="openItem"
      >
        <template #actions="{ item }">
          <DatasetPreviewActions
            :show-delete="false"
            @gallery="openGallery(item)"
          />
        </template>
      </DatasetPreviewCollection>
    </div>

    <DatasetGalleryDialog
      v-model="galleryOpen"
      :title="galleryTitle"
      :content="galleryContent"
      :directory-index="galleryDirectoryIndex"
    />
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { Plus, Search } from "@element-plus/icons-vue";
import { api } from "../api";
import DatasetGalleryDialog from "../components/DatasetGalleryDialog.vue";
import DatasetPreviewActions from "../components/DatasetPreviewActions.vue";
import DatasetPreviewCollection from "../components/DatasetPreviewCollection.vue";
import DatasetViewModeToggle from "../components/DatasetViewModeToggle.vue";
import LibrarySortControls from "../components/LibrarySortControls.vue";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import {
  DATASET_LIBRARY_VIEW_KEY,
  useDatasetViewMode,
} from "../composables/useDatasetViewMode";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatError } from "../lib/formatError";
import { formatLibraryTimestamp } from "../lib/formatLibraryTime.js";
import { libraryThumbSource } from "../lib/previewThumbs";

const router = useRouter();
const rawItems = ref([]);
const loading = ref(false);
const error = ref("");
const query = ref("");
const { viewMode } = useDatasetViewMode(DATASET_LIBRARY_VIEW_KEY);
const {
  sortField,
  sortOrder,
  fieldOptions,
  sortParams,
  toggleSortOrder,
  orderButtonLabel,
} = useLibraryListSort("renga-flow-dataset-list-sort", { kind: "dataset" });

function onToggleSortOrder() {
  toggleSortOrder();
  load();
}

watch([sortField, sortOrder], () => load());
const { galleryOpen, galleryTitle, galleryContent, galleryDirectoryIndex, showFromLibrary } =
  useDatasetGallery();
let searchTimer = null;

const previewItems = computed(() =>
  rawItems.value.map((row) => ({
    key: String(row.id),
    id: row.id,
    title: row.name || `Dataset #${row.id}`,
    subtitle: formatSubtitle(row),
    thumbSource: libraryThumbSource(row.id),
    fallbackText: "DS",
  }))
);

function formatSubtitle(row) {
  const parts = [`#${row.id}`];
  if (row.directory_count != null) {
    parts.push(`${row.directory_count} ${row.directory_count === 1 ? "folder" : "folders"}`);
  }
  if (row.updated_at) parts.push(formatLibraryTimestamp(row.updated_at));
  return parts.join(" · ");
}

function openItem(item) {
  if (item?.id) router.push({ name: "datasets-detail", params: { datasetId: String(item.id) } });
}

function openGallery(item) {
  if (!item?.id) return;
  showFromLibrary({
    id: item.id,
    title: `Gallery — ${item.title}`,
    directoryIndex: null,
  });
}

async function load() {
  loading.value = true;
  error.value = "";
  try {
    const data = await api.searchDatasets({
      q: query.value.trim(),
      page: 1,
      page_size: 100,
      ...sortParams(),
    });
    rawItems.value = data.items || [];
  } catch (e) {
    error.value = formatError(e);
    rawItems.value = [];
  } finally {
    loading.value = false;
  }
}

function scheduleSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(load, 300);
}

onMounted(load);
</script>

<style scoped>
.list-body {
  min-height: 120px;
}
</style>

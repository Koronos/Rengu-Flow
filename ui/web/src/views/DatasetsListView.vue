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

    <LibraryListPage
      :loading="loading"
      :error="error"
      :items="previewItems"
      :view-mode="viewMode"
      :table-actions-column-width="tableActionsWidth"
      empty-description="No datasets yet"
      @item-click="onItemClick"
    >
      <template #empty-action>
        <el-button type="primary" :icon="Plus" @click="router.push({ name: 'datasets-new' })">
          New dataset
        </el-button>
      </template>

      <template #toolbar>
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
        <LibraryViewModeToggle v-model="viewMode" />
      </template>

      <template #actions="{ item }">
        <template v-if="viewMode === 'cards'">
          <LibraryItemOverflowMenu
            :loading="crudBusy"
            @duplicate="duplicateSelected(item.id ?? null)"
            @delete="deleteSelected(item.id ?? null)"
          >
            <el-dropdown-item @click.stop="openGallery(item)">
              <span class="rf-dropdown-item-label">
                <el-icon><Picture /></el-icon>
                <span>Image gallery</span>
              </span>
            </el-dropdown-item>
          </LibraryItemOverflowMenu>
        </template>
        <template v-else>
          <div class="library-list-row-actions" @click.stop>
            <DatasetPreviewActions :show-delete="false" @gallery="openGallery(item)" />
            <LibraryRowCrudButtons
              :loading="crudBusy"
              @duplicate="duplicateSelected(item.id ?? null)"
              @delete="deleteSelected(item.id ?? null)"
            />
          </div>
        </template>
      </template>

      <template #footer>
        <DatasetGalleryDialog
          v-model="galleryOpen"
          :title="galleryTitle"
          :content="galleryContent"
          :directory-index="galleryDirectoryIndex"
          :loading="galleryLoading"
        />
      </template>
    </LibraryListPage>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, watch } from "vue";
import { useRouter } from "vue-router";
import { Picture, Plus, Search } from "@element-plus/icons-vue";
import { api } from "../api";
import DatasetGalleryDialog from "../components/DatasetGalleryDialog.vue";
import DatasetPreviewActions from "../components/DatasetPreviewActions.vue";
import LibraryItemOverflowMenu from "../components/LibraryItemOverflowMenu.vue";
import LibraryListPage from "../components/LibraryListPage.vue";
import LibraryRowCrudButtons from "../components/LibraryRowCrudButtons.vue";
import LibrarySortControls from "../components/LibrarySortControls.vue";
import LibraryViewModeToggle from "../components/LibraryViewModeToggle.vue";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import { useDebouncedLibrarySearch } from "../composables/useDebouncedLibrarySearch";
import { useLibraryCrudActions } from "../composables/useLibraryCrudActions";
import { useLibraryListSelection } from "../composables/useLibraryListSelection";
import {
  DATASET_LIBRARY_VIEW_KEY,
  useDatasetViewMode,
} from "../composables/useDatasetViewMode";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatLibraryTimestamp } from "../lib/formatLibraryTime";
import { libraryThumbSource } from "../lib/previewThumbs";
import type { DatasetSearchItem } from "../types/api";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";

const router = useRouter();
const { viewMode } = useDatasetViewMode(DATASET_LIBRARY_VIEW_KEY);
const {
  sortField,
  sortOrder,
  fieldOptions,
  sortParams,
  toggleSortOrder,
  orderButtonLabel,
} = useLibraryListSort("renga-flow-dataset-list-sort", { kind: "dataset" });
const { rawItems, loading, error, query, load, scheduleSearch } = useDebouncedLibrarySearch(
  api.searchDatasets,
  sortParams
);

function onToggleSortOrder() {
  toggleSortOrder();
  load();
}

watch([sortField, sortOrder], () => load());
const {
  galleryOpen,
  galleryTitle,
  galleryContent,
  galleryDirectoryIndex,
  galleryLoading,
  showFromLibrary,
} = useDatasetGallery();

const basePreviewItems = computed((): DatasetPreviewItem[] =>
  rawItems.value.map((row) => ({
    key: String(row.id),
    id: row.id as string | number,
    title: String(row.name || `Dataset #${row.id}`),
    subtitle: formatSubtitle(row),
    thumbSource: libraryThumbSource(String(row.id)),
    fallbackText: "DS",
  }))
);

const { selectedId, previewItems, selectItem, clearSelection } =
  useLibraryListSelection(basePreviewItems);

const {
  busy: crudBusy,
  duplicateSelected,
  deleteSelected,
} = useLibraryCrudActions("dataset", {
  router,
  onDeleted: () => {
    clearSelection();
    load();
  },
});

function formatSubtitle(row: DatasetSearchItem): string {
  const parts = [`#${row.id}`];
  if (row.directory_count != null) {
    parts.push(`${row.directory_count} ${row.directory_count === 1 ? "folder" : "folders"}`);
  }
  if (row.updated_at) parts.push(formatLibraryTimestamp(row.updated_at));
  return parts.join(" · ");
}

const tableActionsWidth = computed(() => (viewMode.value === "table" ? 248 : 220));

function onItemClick(item: DatasetPreviewItem) {
  selectItem(item);
  openItem(item);
}

function openItem(item: DatasetPreviewItem) {
  if (item?.id) router.push({ name: "datasets-detail", params: { datasetId: String(item.id) } });
}

function openGallery(item: DatasetPreviewItem) {
  if (!item?.id) return;
  showFromLibrary({
    id: item.id,
    title: `Gallery — ${item.title}`,
    directoryIndex: null,
  });
}

onMounted(() => load());
</script>

<template>
  <ImportTomlOverlay ref="importOverlay" @import="importConfigFile">
    <div class="configs-page page-shell">
      <div class="page-head">
        <div class="page-head-text">
          <p class="page-subtitle">Model training TOML library</p>
        </div>
        <div class="page-head-actions">
          <el-button type="primary" :icon="Plus" @click="router.push({ name: 'configs-new' })">
            New config
          </el-button>
          <el-button @click="triggerImport">Import TOML…</el-button>
        </div>
      </div>

      <LibraryListPage
        :loading="loading"
        :error="error"
        :items="previewItems"
        :view-mode="viewMode"
        :table-actions-column-width="tableActionsWidth"
        empty-description="No configs yet"
        @item-click="onItemClick"
      >
        <template #empty-action>
          <el-button type="primary" :icon="Plus" @click="router.push({ name: 'configs-new' })">
            New config
          </el-button>
        </template>

        <template #banner>
          <PickForJobBanner v-if="pickForJob" variant="list" @cancel="cancelPick" />
        </template>

        <template #toolbar>
          <el-input
            v-model="query"
            clearable
            placeholder="Search by ID, run name, model, or dataset…"
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
              <el-dropdown-item v-if="pickForJob" @click.stop="openItem(item)">
                <span class="rf-dropdown-item-label">
                  <el-icon><FolderOpened /></el-icon>
                  <span>Open</span>
                </span>
              </el-dropdown-item>
              <el-dropdown-item v-else @click.stop="startRun(item)">
                <span class="rf-dropdown-item-label">
                  <el-icon><VideoPlay /></el-icon>
                  <span>Run training job</span>
                </span>
              </el-dropdown-item>
            </LibraryItemOverflowMenu>
          </template>
          <template v-else>
            <div class="library-list-row-actions" @click.stop>
              <el-tooltip v-if="pickForJob" content="Open" :show-after="300">
                <el-button
                  size="small"
                  circle
                  type="primary"
                  :icon="FolderOpened"
                  @click.stop="openItem(item)"
                />
              </el-tooltip>
              <el-tooltip v-else content="Run training job" :show-after="300">
                <el-button
                  size="small"
                  circle
                  type="primary"
                  :icon="VideoPlay"
                  @click.stop="startRun(item)"
                />
              </el-tooltip>
              <LibraryRowCrudButtons
                :loading="crudBusy"
                @duplicate="duplicateSelected(item.id ?? null)"
                @delete="deleteSelected(item.id ?? null)"
              />
            </div>
          </template>
        </template>
      </LibraryListPage>
    </div>
  </ImportTomlOverlay>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { FolderOpened, Plus, Search, VideoPlay } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import ImportTomlOverlay from "../components/ImportTomlOverlay.vue";
import LibraryItemOverflowMenu from "../components/LibraryItemOverflowMenu.vue";
import LibraryListPage from "../components/LibraryListPage.vue";
import LibraryRowCrudButtons from "../components/LibraryRowCrudButtons.vue";
import LibrarySortControls from "../components/LibrarySortControls.vue";
import LibraryViewModeToggle from "../components/LibraryViewModeToggle.vue";
import PickForJobBanner from "../components/PickForJobBanner.vue";
import { useDatasetViewMode } from "../composables/useDatasetViewMode";
import { useDebouncedLibrarySearch } from "../composables/useDebouncedLibrarySearch";
import { useImportConfigToml } from "../composables/useImportConfigToml";
import { useLibraryCrudActions } from "../composables/useLibraryCrudActions";
import { useLibraryListSelection } from "../composables/useLibraryListSelection";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatLibraryTimestamp } from "../lib/formatLibraryTime";
import { datasetRefToThumbSource } from "../lib/previewThumbs";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";
import { redirectToStoredJobConfig } from "../lib/jobConfigRedirect";
import type { ConfigSearchItem } from "../types/api";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";
import type ImportTomlOverlayType from "../components/ImportTomlOverlay.vue";

const CONFIG_LIBRARY_VIEW_KEY = "rengu-flow-config-library-view";

const route = useRoute();
const router = useRouter();
const importOverlay = ref<InstanceType<typeof ImportTomlOverlayType> | null>(null);
const { viewMode } = useDatasetViewMode(CONFIG_LIBRARY_VIEW_KEY);
const {
  sortField,
  sortOrder,
  fieldOptions,
  sortParams,
  toggleSortOrder,
  orderButtonLabel,
} = useLibraryListSort("rengu-flow-config-list-sort", { kind: "config" });
const { rawItems, loading, error, query, load, scheduleSearch } = useDebouncedLibrarySearch(
  api.searchConfigs,
  sortParams
);
const { importConfigFile } = useImportConfigToml();

const pickForJob = computed(() => route.query.pick === "job");

const basePreviewItems = computed((): DatasetPreviewItem[] =>
  rawItems.value.map((row) => ({
    key: String(row.id),
    id: row.id as string | number,
    title: configTitle(row),
    subtitle: formatSubtitle(row),
    thumbSource: datasetRefToThumbSource(row.dataset_ref),
    fallbackText: configFallback(row),
  }))
);

const { selectedId, previewItems, selectItem, clearSelection } =
  useLibraryListSelection(basePreviewItems);

const {
  busy: crudBusy,
  duplicateSelected,
  deleteSelected,
} = useLibraryCrudActions("config", {
  router,
  onDeleted: () => {
    clearSelection();
    load();
  },
});

function configTitle(row: ConfigSearchItem): string {
  if (row.run_name) return String(row.run_name);
  return `Config #${row.id}`;
}

function configFallback(row: ConfigSearchItem): string {
  const mt = row.model_type;
  if (typeof mt === "string" && mt.length >= 2) return mt.slice(0, 2).toUpperCase();
  return "CF";
}

function formatSubtitle(row: ConfigSearchItem): string {
  const parts = [`#${row.id}`];
  const runName =
    typeof row.run_name === "string" && row.run_name.trim() ? row.run_name.trim() : "";
  if (runName && configTitle(row) !== runName) parts.push(runName);
  if (row.model_type) parts.push(row.model_type);
  if (row.dataset_ref) parts.push(row.dataset_ref);
  if (row.updated_at) parts.push(formatLibraryTimestamp(row.updated_at));
  return parts.join(" · ");
}

const tableActionsWidth = computed(() => (viewMode.value === "table" ? 280 : 252));

function onItemClick(item: DatasetPreviewItem): void {
  selectItem(item);
  openItem(item);
}

function openItem(item: DatasetPreviewItem): void {
  if (item?.id == null) return;
  const query = pickForJob.value ? { pick: "job" } : {};
  router.push({
    name: "configs-detail",
    params: { configId: String(item.id) },
    query,
  });
}

function startRun(item: DatasetPreviewItem): void {
  if (item?.id == null) return;
  setJobConfigId(String(item.id));
  router.push({ name: "jobs" });
  ElMessage.success(`"${item.id}" selected — configure GPUs on Runs`);
}

function cancelPick() {
  router.push({ name: "jobs" });
}

function onToggleSortOrder() {
  toggleSortOrder();
  load();
}

watch([sortField, sortOrder], () => load());

function triggerImport() {
  importOverlay.value?.openFilePicker?.();
}

onMounted(async () => {
  if (pickForJob.value) {
    const redirected = await redirectToStoredJobConfig(
      router,
      getJobConfigId(),
      (id) => api.getConfig(id)
    );
    if (redirected) return;
  }
  load();
});
</script>

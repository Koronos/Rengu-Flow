<template>
  <ImportTomlOverlay ref="importOverlay" @import="onImportFile">
    <div class="configs-page page-shell">
      <div class="page-head">
        <div class="page-head-text">
          <p class="page-subtitle">Model training TOML library</p>
        </div>
        <div class="page-head-actions">
          <el-button @click="triggerImport">Import TOML…</el-button>
          <el-button type="primary" :icon="Plus" @click="router.push({ name: 'configs-new' })">
            New config
          </el-button>
        </div>
      </div>

      <el-alert
        v-if="pickForJob"
        type="warning"
        :closable="false"
        show-icon
        class="mb-12 pick-banner"
      >
        <template #title>Choosing config for a training job</template>
        Open a config below, edit if needed, validate, then click
        <strong>Use for training job</strong> in the editor. You can also
        <el-button type="primary" link @click="cancelPick">return to Runs</el-button>
        without selecting.
      </el-alert>

      <div class="page-toolbar">
        <el-input
          v-model="query"
          clearable
          placeholder="Search by ID, model, or dataset…"
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
        <el-empty v-if="!loading && !previewItems.length" description="No configs yet" :image-size="64" />
        <DatasetPreviewCollection
          v-else
          :items="previewItems"
          :view-mode="viewMode"
          table-subtitle-label="Summary"
          @item-click="openItem"
        >
          <template #actions="{ item }">
            <el-button
              v-if="pickForJob"
              type="primary"
              size="small"
              plain
              @click.stop="openItem(item)"
            >
              Open
            </el-button>
            <el-button
              v-else
              type="primary"
              size="small"
              plain
              @click.stop="startRun(item)"
            >
              Run
            </el-button>
          </template>
        </DatasetPreviewCollection>
      </div>
    </div>
  </ImportTomlOverlay>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { Plus, Search } from "@element-plus/icons-vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import DatasetPreviewCollection from "../components/DatasetPreviewCollection.vue";
import DatasetViewModeToggle from "../components/DatasetViewModeToggle.vue";
import ImportTomlOverlay from "../components/ImportTomlOverlay.vue";
import LibrarySortControls from "../components/LibrarySortControls.vue";
import { useDatasetViewMode } from "../composables/useDatasetViewMode";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatError } from "../lib/formatError";
import { formatLibraryTimestamp } from "../lib/formatLibraryTime";
import { getJobConfigId, setJobConfigId } from "../lib/jobConfigPick";
import type { JsonRecord } from "../types/runtime";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";
import type ImportTomlOverlayType from "../components/ImportTomlOverlay.vue";

const CONFIG_LIBRARY_VIEW_KEY = "renga-flow-config-library-view";

const route = useRoute();
const router = useRouter();
const rawItems = ref<JsonRecord[]>([]);
const loading = ref(false);
const error = ref("");
const query = ref("");
const importOverlay = ref<InstanceType<typeof ImportTomlOverlayType> | null>(null);
const { viewMode } = useDatasetViewMode(CONFIG_LIBRARY_VIEW_KEY);
const {
  sortField,
  sortOrder,
  fieldOptions,
  sortParams,
  toggleSortOrder,
  orderButtonLabel,
} = useLibraryListSort("renga-flow-config-list-sort", { kind: "config" });

const pickForJob = computed(() => route.query.pick === "job");

let searchTimer: ReturnType<typeof setTimeout> | undefined;

const previewItems = computed((): DatasetPreviewItem[] =>
  rawItems.value.map((row) => ({
    key: String(row.id),
    id: row.id as string | number,
    title: configTitle(row),
    subtitle: formatSubtitle(row),
    fallbackText: configFallback(row),
  }))
);

function configTitle(row) {
  if (row.run_name) return String(row.run_name);
  return `Config #${row.id}`;
}

function configFallback(row) {
  const mt = row.model_type;
  if (typeof mt === "string" && mt.length >= 2) return mt.slice(0, 2).toUpperCase();
  return "CF";
}

function formatSubtitle(row) {
  const parts = [`#${row.id}`];
  if (row.model_type) parts.push(row.model_type);
  if (row.dataset_ref) parts.push(row.dataset_ref);
  if (row.updated_at) parts.push(formatLibraryTimestamp(row.updated_at));
  return parts.join(" · ");
}

function openItem(item) {
  if (item?.id == null) return;
  const query = pickForJob.value ? { pick: "job" } : {};
  router.push({
    name: "configs-detail",
    params: { configId: String(item.id) },
    query,
  });
}

function startRun(item) {
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

async function load() {
  loading.value = true;
  error.value = "";
  try {
    const data = (await api.searchConfigs({
      q: query.value.trim(),
      page: 1,
      page_size: 100,
      ...sortParams(),
    })) as { items?: JsonRecord[] };
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

function triggerImport() {
  importOverlay.value?.openFilePicker?.();
}

async function onImportFile(file) {
  try {
    const text = await file.text();
    const base = file.name.replace(/\.toml$/i, "") || "imported";
    const r = (await api.importConfig(text, base)) as { id: string };
    ElMessage.success(`Imported as ${r.id}`);
    router.push({ name: "configs-detail", params: { configId: String(r.id) } });
  } catch (e) {
    ElMessage.error(formatError(e));
  }
}

onMounted(async () => {
  if (pickForJob.value) {
    const stored = getJobConfigId();
    if (stored) {
      try {
        await api.getConfig(stored);
        router.replace({
          name: "configs-detail",
          params: { configId: stored },
          query: { pick: "job" },
        });
        return;
      } catch {
        /* stored config may have been deleted */
      }
    }
  }
  load();
});
</script>

<style scoped>
.list-body {
  min-height: 120px;
}
.pick-banner {
  line-height: 1.5;
}
</style>

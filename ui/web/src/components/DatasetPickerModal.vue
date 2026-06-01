<template>
  <el-dialog
    :model-value="modelValue"
    :title="multiple ? 'Choose datasets' : 'Choose dataset'"
    width="92%"
    class="dataset-picker-dialog"
    append-to-body
    align-center
    destroy-on-close
    @update:model-value="$emit('update:modelValue', $event)"
    @open="loadItems"
  >
    <div v-loading="loading" class="picker-body">
      <div class="picker-head">
        <el-input
          v-model="filterText"
          clearable
          placeholder="Filter by name or ID…"
          class="picker-filter"
        />
        <DatasetViewModeToggle v-model="viewMode" />
      </div>
      <p v-if="multiple && selectedPaths.length" class="picker-selected">
        Selected: {{ selectedPaths.length }}
      </p>
      <el-empty v-if="!loading && !filteredItems.length" description="No datasets found" />
      <DatasetPreviewCollection
        v-else
        :items="filteredItems"
        :view-mode="viewMode"
        scrollable
        show-check
        table-title-label="Name"
        table-subtitle-label="Library ref"
        :table-actions-column-width="148"
        @item-click="onItemClick"
      >
        <template #actions="{ item }">
          <DatasetPreviewActions
            :show-delete="false"
            :gallery-disabled="!item.libraryId"
            @gallery="openGallery(item)"
          >
            <el-tooltip v-if="item.libraryId" content="Edit dataset (new tab)" :show-after="300">
              <el-button size="small" circle :icon="Edit" @click="openEdit(item)" />
            </el-tooltip>
          </DatasetPreviewActions>
        </template>
      </DatasetPreviewCollection>
      <p v-if="!multiple" class="picker-compose-hint">
        Need multiple folders in one TOML?
        <router-link to="/datasets">Compose datasets</router-link>
        in the Datasets library.
      </p>
    </div>
    <template #footer>
      <el-button @click="$emit('update:modelValue', false)">Cancel</el-button>
      <el-button
        v-if="multiple"
        type="primary"
        :disabled="!selectedPaths.length"
        @click="confirmMulti"
      >
        Add selected
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { Edit } from "@element-plus/icons-vue";
import { ElLoadingDirective } from "element-plus";
import { api } from "../api";
import DatasetPreviewActions from "./DatasetPreviewActions.vue";
import DatasetPreviewCollection from "./DatasetPreviewCollection.vue";
import DatasetViewModeToggle from "./DatasetViewModeToggle.vue";
import { useDatasetFormModal } from "../composables/useDatasetFormModal";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import { useDatasetViewMode } from "../composables/useDatasetViewMode";
import { canonicalDatasetRef, formatDatasetLibraryRef } from "../lib/datasetLibraryRef";
import { cacheDatasetDisplayLabel } from "../lib/resolveDatasetLabels";
import { libraryThumbSource } from "../lib/previewThumbs";
import type { DatasetSearchItem } from "../types/api";
import type { DatasetPreviewItem } from "./DatasetPreviewCollection.vue";

type DatasetPickerItem = DatasetPreviewItem & {
  path: string;
  libraryId: string | null;
};

interface DatasetPickerModalProps {
  modelValue: boolean;
  multiple: boolean;
  selected: string | string[];
}

const props = withDefaults(defineProps<DatasetPickerModalProps>(), {
  modelValue: false,
  multiple: false,
  selected: "",
});

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "select", value: string): void;
  (e: "select-multiple", value: string[]): void;
}>();
const vLoading = ElLoadingDirective;
const datasetModal = useDatasetFormModal();

const loading = ref(false);
const filterText = ref("");
const items = ref<DatasetPickerItem[]>([]);
const selectedPaths = ref<string[]>([]);
const { viewMode } = useDatasetViewMode("rengu-flow-dataset-picker-view", "table");
const { showFromLibrary } = useDatasetGallery();

const filteredItems = computed(() => {
  const q = filterText.value.trim().toLowerCase();
  const list = q
    ? items.value.filter(
        (item) =>
          (item.title || "").toLowerCase().includes(q) ||
          (item.subtitle || "").toLowerCase().includes(q) ||
          String(item.id ?? "").includes(q)
      )
    : items.value;
  return list.map((item) => ({
    ...item,
    active: isSelected(item.path),
  }));
});

function formatRowSubtitle(row: DatasetSearchItem): string {
  const parts: string[] = [formatDatasetLibraryRef(row.id, row.name)];
  if (row.directory_count != null) {
    parts.push(
      `${row.directory_count} ${row.directory_count === 1 ? "folder" : "folders"}`
    );
  }
  return parts.join(" · ");
}

async function loadItems() {
  loading.value = true;
  try {
    const { items: rows } = await api.searchDatasets({
      q: "",
      page: 1,
      page_size: 100,
      sort: "id",
      order: "desc",
    });
    items.value = (rows ?? []).map((row): DatasetPickerItem => {
      const libraryId = String(row.id);
      const name = String(row.name || `Dataset #${row.id}`);
      const path = formatDatasetLibraryRef(row.id, name);
      const display = `${name} (#${row.id})`;
      cacheDatasetDisplayLabel(path, display);
      return {
        key: path,
        id: row.id,
        path,
        libraryId,
        title: name,
        subtitle: formatRowSubtitle(row),
        thumbSource: libraryThumbSource(libraryId),
        fallbackText: "DS",
      };
    });
  } catch {
    items.value = [];
  } finally {
    loading.value = false;
  }
}

function isSelected(path: string): boolean {
  const key = canonicalDatasetRef(path);
  if (props.multiple) {
    return selectedPaths.value.some((p) => canonicalDatasetRef(p) === key);
  }
  return canonicalDatasetRef(props.selected) === key;
}

function onItemClick(item: DatasetPickerItem) {
  toggle(item.path);
}

function openGallery(item: DatasetPickerItem) {
  if (!item.libraryId) return;
  showFromLibrary({
    id: item.libraryId,
    title: `Gallery — ${item.title}`,
    directoryIndex: null,
  });
}

function openEdit(item: DatasetPickerItem) {
  if (!item.libraryId) return;
  datasetModal.openEdit(item.libraryId, { onSaved: () => loadItems() });
}

function toggle(path: string) {
  if (props.multiple) {
    const key = canonicalDatasetRef(path);
    const idx = selectedPaths.value.findIndex((p) => canonicalDatasetRef(p) === key);
    if (idx >= 0) selectedPaths.value.splice(idx, 1);
    else selectedPaths.value.push(path);
    return;
  }
  emit("select", path);
  emit("update:modelValue", false);
}

function confirmMulti() {
  emit("select-multiple", [...selectedPaths.value]);
  emit("update:modelValue", false);
}

watch(
  () => props.modelValue,
  (open) => {
    if (!open) return;
    if (props.multiple) {
      selectedPaths.value = Array.isArray(props.selected) ? [...props.selected] : [];
    }
    filterText.value = "";
  }
);
</script>

<style scoped>
:global(.dataset-picker-dialog) {
  max-width: 920px;
}
.picker-body {
  min-height: 200px;
}
.picker-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.picker-filter {
  flex: 1;
  min-width: 160px;
}
.picker-selected {
  margin: 0 0 8px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.picker-compose-hint {
  margin: 12px 0 0;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.picker-compose-hint a {
  color: var(--el-color-primary);
  text-decoration: none;
}
.picker-compose-hint a:hover {
  text-decoration: underline;
}
</style>

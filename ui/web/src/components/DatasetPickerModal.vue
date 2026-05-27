<template>
  <el-dialog
    :model-value="modelValue"
    :title="multiple ? 'Choose datasets' : 'Choose dataset'"
    width="92%"
    style="max-width: 720px"
    destroy-on-close
    @update:model-value="$emit('update:modelValue', $event)"
    @open="loadItems"
  >
    <div v-loading="loading" class="picker-body">
      <div class="picker-head">
        <el-input
          v-model="filterText"
          clearable
          placeholder="Filter datasets…"
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
        @item-click="onItemClick"
      >
        <template #actions="{ item }">
          <DatasetPreviewActions
            :show-delete="false"
            :gallery-disabled="!item.libraryId"
            @gallery="openGallery(item)"
          />
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

    <DatasetGalleryDialog
      v-model="galleryOpen"
      :title="galleryTitle"
      :content="galleryContent"
      :directory-index="galleryDirectoryIndex"
    />
  </el-dialog>
</template>

<script setup>
import { computed, ref, watch } from "vue";
import { api } from "../api";
import DatasetGalleryDialog from "./DatasetGalleryDialog.vue";
import DatasetPreviewActions from "./DatasetPreviewActions.vue";
import DatasetPreviewCollection from "./DatasetPreviewCollection.vue";
import DatasetViewModeToggle from "./DatasetViewModeToggle.vue";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import { useDatasetViewMode } from "../composables/useDatasetViewMode";
import { canonicalDatasetRef } from "../lib/datasetLibraryRef";
import { libraryThumbSource } from "../lib/previewThumbs";

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  multiple: { type: Boolean, default: false },
  selected: { type: [String, Array], default: "" },
});

const emit = defineEmits(["update:modelValue", "select", "select-multiple"]);

const loading = ref(false);
const filterText = ref("");
const items = ref([]);
const selectedPaths = ref([]);
const { viewMode } = useDatasetViewMode("renga-flow-dataset-picker-view");
const { galleryOpen, galleryTitle, galleryContent, galleryDirectoryIndex, showFromLibrary } =
  useDatasetGallery();

const filteredItems = computed(() => {
  const q = filterText.value.trim().toLowerCase();
  const list = q
    ? items.value.filter(
        (item) =>
          item.title.toLowerCase().includes(q) || item.subtitle.toLowerCase().includes(q)
      )
    : items.value;
  return list.map((item) => ({
    ...item,
    active: isSelected(item.path),
  }));
});

function parseLibraryId(path) {
  const key = canonicalDatasetRef(path);
  const m = key.match(/^renga-flow-dataset:(\d+)$/);
  return m ? m[1] : null;
}

async function loadItems() {
  loading.value = true;
  try {
    const schema = await api.getSchema();
    const picker = schema?.registries?.dataset_paths || [];
    items.value = picker.map((entry) => {
        const libraryId = parseLibraryId(entry.path);
      return {
        key: entry.path,
        path: entry.path,
        libraryId,
        title: entry.label || entry.id || entry.path,
        subtitle: entry.path,
        thumbSource: libraryId ? libraryThumbSource(libraryId) : null,
        fallbackText: "DS",
      };
    });
  } catch {
    items.value = [];
  } finally {
    loading.value = false;
  }
}

function isSelected(path) {
  const key = canonicalDatasetRef(path);
  if (props.multiple) {
    return selectedPaths.value.some((p) => canonicalDatasetRef(p) === key);
  }
  return canonicalDatasetRef(props.selected) === key;
}

function onItemClick(item) {
  toggle(item.path);
}

function openGallery(item) {
  if (!item.libraryId) return;
  showFromLibrary({
    id: item.libraryId,
    title: `Gallery — ${item.title}`,
    directoryIndex: null,
  });
}

function toggle(path) {
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

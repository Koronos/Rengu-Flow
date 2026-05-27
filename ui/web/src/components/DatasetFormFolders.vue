<template>
  <div class="dataset-folders">
    <el-alert
      v-if="uiNotes.length"
      type="info"
      title="Some TOML keys are not shown in the form"
      show-icon
      :closable="false"
      class="mb-12"
    >
      <ul class="ui-notes">
        <li v-for="(note, i) in uiNotes" :key="i">{{ note }}</li>
      </ul>
    </el-alert>

    <p class="tab-intro">
      One row per <code>[[directory]]</code> table in the dataset TOML (image folder path and optional overrides).
    </p>

    <div class="folders-toolbar">
      <el-input
        v-model="query"
        clearable
        placeholder="Search [[directory]] by path…"
        class="folders-search"
        :prefix-icon="Search"
      />
      <div class="folders-toolbar-end">
        <DatasetViewModeToggle v-model="viewMode" />
        <el-button type="primary" :icon="Plus" @click="openAdd">Add directory</el-button>
      </div>
    </div>

    <p v-if="directories.length" class="folders-count">
      <template v-if="filteredEntries.length !== directories.length">
        {{ filteredEntries.length }} of {{ directories.length }} directories
      </template>
      <template v-else>
        {{ directories.length }} {{ directories.length === 1 ? "directory" : "directories" }}
      </template>
    </p>

    <DatasetPreviewCollection
      v-if="previewItems.length"
      :items="previewItems"
      :view-mode="viewMode"
      dense
      scrollable
      table-title-label="Directory"
      table-subtitle-label="Path"
      @item-click="onPreviewClick"
    >
      <template #tags="{ item }">
        <el-tag v-if="item.warning" size="small" type="warning">No path</el-tag>
        <el-tag
          v-if="item.dir.num_repeats && item.dir.num_repeats !== 1"
          size="small"
          type="info"
        >
          ×{{ item.dir.num_repeats }}
        </el-tag>
        <el-tag v-if="item.overrideCount" size="small">
          {{ item.overrideCount }} override{{ item.overrideCount === 1 ? "" : "s" }}
        </el-tag>
      </template>
      <template #actions="{ item }">
        <DatasetPreviewActions
          :gallery-disabled="item.warning"
          delete-title="Remove directory"
          @gallery="openGallery(item)"
          @delete="removeAt(item.index)"
        />
      </template>
    </DatasetPreviewCollection>

    <el-empty
      v-else-if="directories.length && query.trim()"
      description="No directories match your search"
      :image-size="56"
    />
    <el-empty v-else description="No [[directory]] entries yet" :image-size="56">
      <el-button type="primary" :icon="Plus" @click="openAdd">Add directory</el-button>
    </el-empty>

    <DatasetFolderDialog
      v-model="dialogOpen"
      :schema="schema"
      :entry="dialogEntry"
      :edit-index="dialogIndex"
      @save="onDialogSave"
    />

    <DatasetGalleryDialog
      v-model="galleryOpen"
      :title="galleryTitle"
      :content="galleryContent"
      :directory-index="galleryDirectoryIndex"
    />
  </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { storeToRefs } from "pinia";
import { ElMessageBox } from "element-plus";
import { Plus, Search } from "@element-plus/icons-vue";
import DatasetFolderDialog from "./DatasetFolderDialog.vue";
import DatasetGalleryDialog from "./DatasetGalleryDialog.vue";
import DatasetPreviewActions from "./DatasetPreviewActions.vue";
import DatasetPreviewCollection from "./DatasetPreviewCollection.vue";
import DatasetViewModeToggle from "./DatasetViewModeToggle.vue";
import { useDatasetGallery } from "../composables/useDatasetGallery";
import {
  DATASET_DIRECTORY_VIEW_KEY,
  useDatasetViewMode,
} from "../composables/useDatasetViewMode";
import {
  basenameFromPath,
  countDirectoryOverrides,
  emptyDirectoryRow,
} from "../lib/datasetDirectoryForm";
import { pathThumbSource } from "../lib/previewThumbs";
import { useDatasetEditorStore } from "../stores/datasetEditor";

const editor = useDatasetEditorStore();
const { form, schema, uiNotes, content } = storeToRefs(editor);
const { viewMode } = useDatasetViewMode(DATASET_DIRECTORY_VIEW_KEY);
const {
  galleryOpen,
  galleryTitle,
  galleryContent,
  galleryDirectoryIndex,
  showFromContent,
} = useDatasetGallery();

const query = ref("");
const dialogOpen = ref(false);
const dialogIndex = ref(-1);
const dialogEntry = ref(null);

const directories = computed(() => {
  const dirs = form.value?._directories;
  return Array.isArray(dirs) ? dirs : [];
});

const filteredEntries = computed(() => {
  const q = query.value.trim().toLowerCase();
  return directories.value
    .map((dir, index) => ({ dir, index }))
    .filter(({ dir, index }) => {
      if (!q) return true;
      const path = (dir.path || "").toLowerCase();
      const title = displayTitle(dir, index).toLowerCase();
      return path.includes(q) || title.includes(q) || (isPathEmpty(dir) && "no path".includes(q));
    });
});

const previewItems = computed(() =>
  filteredEntries.value.map(({ dir, index }) => ({
    key: folderKey(dir, index),
    index,
    dir,
    title: displayTitle(dir, index),
    subtitle: pathLabel(dir),
    thumbSource: pathThumbSource(dir.path || ""),
    warning: isPathEmpty(dir),
    stacked: true,
    overrideCount: countDirectoryOverrides(dir),
    fallbackText: isPathEmpty(dir)
      ? "—"
      : basenameFromPath(dir.path).slice(0, 2).toUpperCase() || "…",
  }))
);

function patchDirectories(nextDirs) {
  editor.patchDirectories(nextDirs);
}

function isPathEmpty(dir) {
  return !(dir.path || "").trim();
}

function folderKey(dir, index) {
  return `dir-${index}:${(dir.path || "").trim()}`;
}

function displayTitle(dir, index) {
  if (isPathEmpty(dir)) {
    return `Directory #${index + 1}`;
  }
  const base = basenameFromPath(dir.path);
  return base || dir.path;
}

function pathLabel(dir) {
  if (isPathEmpty(dir)) {
    return "path not set in TOML";
  }
  return dir.path;
}

function onPreviewClick(item) {
  openEdit(item.index);
}

function openGallery(item) {
  showFromContent({
    title: `Gallery — ${item.title}`,
    content: content.value,
    directoryIndex: item.index,
  });
}

function openAdd() {
  dialogIndex.value = -1;
  dialogEntry.value = emptyDirectoryRow();
  dialogOpen.value = true;
}

function openEdit(index) {
  dialogIndex.value = index;
  dialogEntry.value = directories.value[index];
  dialogOpen.value = true;
}

function onDialogSave({ entry, index }) {
  const next = [...directories.value];
  if (index >= 0) {
    next[index] = entry;
  } else {
    next.push(entry);
  }
  patchDirectories(next);
}

async function removeAt(index) {
  const dir = directories.value[index];
  const label = displayTitle(dir, index);
  try {
    await ElMessageBox.confirm(`Remove "${label}" from the dataset?`, "Remove directory", {
      type: "warning",
      confirmButtonText: "Remove",
    });
  } catch {
    return;
  }
  const next = directories.value.filter((_, i) => i !== index);
  patchDirectories(next);
}
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
.ui-notes {
  margin: 4px 0 0;
  padding-left: 18px;
  font-size: 13px;
}
.tab-intro {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  line-height: 1.45;
}
.tab-intro code {
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
}
.folders-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.folders-search {
  flex: 1;
  min-width: 160px;
}
.folders-toolbar-end {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
  margin-left: auto;
}
.folders-count {
  margin: 0 0 10px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
@media (max-width: 640px) {
  .folders-toolbar-end {
    width: 100%;
    margin-left: 0;
    justify-content: space-between;
  }
}
</style>

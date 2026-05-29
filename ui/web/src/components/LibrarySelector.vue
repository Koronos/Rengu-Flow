<template>
  <div class="library-selector">
    <div class="library-selector__row">
      <el-autocomplete
        v-model="searchText"
        class="library-selector__search"
        clearable
        :placeholder="searchPlaceholder"
        :fetch-suggestions="fetchSuggestions"
        :trigger-on-focus="true"
        value-key="id"
        @select="onSuggestSelect"
      >
        <template #default="{ item }">
          <div class="suggest-row">
            <span class="suggest-id">{{ item.id }}</span>
            <span v-if="item.hint" class="suggest-hint">{{ item.hint }}</span>
          </div>
        </template>
      </el-autocomplete>
      <el-button :icon="Search" @click="openBrowser">Browse library</el-button>
      <el-button :disabled="!modelValue" @click="emitAction('duplicate', modelValue)">
        Duplicate
      </el-button>
      <el-button
        type="danger"
        plain
        :disabled="!modelValue"
        @click="emitAction('delete', modelValue)"
      >
        Delete
      </el-button>
      <el-dropdown v-if="modelValue" trigger="click" @command="onAction">
        <el-button>
          More
          <el-icon class="el-icon--right"><ArrowDown /></el-icon>
        </el-button>
        <template #dropdown>
          <el-dropdown-item command="open">Open in editor</el-dropdown-item>
          <el-dropdown-item command="copy-new">New from copy…</el-dropdown-item>
          <el-dropdown-item v-if="kind === 'config' && pickForJob" command="use-job">
            Use for training job
          </el-dropdown-item>
          <el-dropdown-item v-if="kind === 'config'" command="start-job">
            Start training job
          </el-dropdown-item>
        </template>
      </el-dropdown>
    </div>
    <el-text v-if="modelValue" type="info" size="small" class="library-selector__active">
      Active: <strong>{{ modelValue }}</strong>
      <span v-if="activeHint"> · {{ activeHint }}</span>
    </el-text>

    <el-dialog
      v-model="browserOpen"
      :title="browserTitle"
      width="92%"
      style="max-width: 720px"
      destroy-on-close
      @open="onBrowserOpen"
    >
      <div class="browser-toolbar">
        <el-input
          v-model="browserQuery"
          clearable
          :placeholder="searchPlaceholder"
        >
          <template #append>
            <el-button :icon="Search" @click="loadBrowser(1)" />
          </template>
        </el-input>
        <LibrarySortControls
          v-model:sort-field="sortField"
          :sort-order="sortOrder"
          :field-options="fieldOptions"
          :order-button-label="orderButtonLabel"
          @toggle-order="onToggleBrowserSort"
        />
      </div>

      <el-table
        v-loading="browserLoading"
        :data="browserItems"
        size="small"
        stripe
        highlight-current-row
        class="browser-table"
        @row-click="onRowClick"
      >
        <el-table-column prop="id" label="ID" min-width="140" show-overflow-tooltip />
        <el-table-column v-if="kind === 'config'" prop="model_type" label="Model" width="100" />
        <el-table-column
          v-if="kind === 'config'"
          prop="dataset_ref"
          label="Dataset"
          min-width="120"
          show-overflow-tooltip
        />
        <el-table-column
          v-if="kind === 'dataset'"
          prop="directory_count"
          label="Dirs"
          width="64"
        />
        <el-table-column label="Created" width="132">
          <template #default="{ row }">
            {{ formatLibraryTimestamp(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="Updated" width="132">
          <template #default="{ row }">
            {{ formatLibraryTimestamp(row.updated_at) }}
          </template>
        </el-table-column>
        <el-table-column label="" width="220" fixed="right">
          <template #default="{ row }">
            <el-button-group size="small">
              <el-button @click.stop="selectAndOpen(row.id)">Open</el-button>
              <el-button @click.stop="emitAction('duplicate', row.id)">Copy</el-button>
              <el-button
                v-if="kind === 'config'"
                type="primary"
                plain
                @click.stop="emitAction('start-job', row.id)"
              >
                Train
              </el-button>
              <el-button type="danger" plain @click.stop="emitAction('delete', row.id)">
                Del
              </el-button>
            </el-button-group>
          </template>
        </el-table-column>
      </el-table>

      <div class="browser-footer">
        <el-text type="info" size="small">
          {{ browserTotal }} total
        </el-text>
        <el-pagination
          v-model:current-page="browserPage"
          v-model:page-size="browserPageSize"
          :total="browserTotal"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          small
          background
          @current-change="loadBrowser"
          @size-change="onPageSizeChange"
        />
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { ArrowDown, Search } from "@element-plus/icons-vue";
import { ElLoadingDirective } from "element-plus";
import { api } from "../api";
import { useLibraryListSort } from "../composables/useLibraryListSort";
import { formatLibraryTimestamp } from "../lib/formatLibraryTime";
import type { ConfigSearchItem, DatasetSearchItem } from "../types/api";
import LibrarySortControls from "./LibrarySortControls.vue";

type LibraryKind = "config" | "dataset";
type LibraryRow = ConfigSearchItem | DatasetSearchItem;
const vLoading = ElLoadingDirective;

const props = defineProps({
  kind: { type: String as () => LibraryKind, required: true }, // "config" | "dataset"
  modelValue: { type: String, default: null },
  pickForJob: { type: Boolean, default: false },
  activeHint: { type: String, default: "" },
});

const emit = defineEmits([
  "update:modelValue",
  "open",
  "duplicate",
  "delete",
  "start-job",
  "use-for-job",
  "new-from-copy",
]);

const searchText = ref("");
const browserOpen = ref(false);
const browserQuery = ref("");
const browserItems = ref<LibraryRow[]>([]);
const browserTotal = ref(0);
const browserPage = ref(1);
const browserPageSize = ref(20);
const browserLoading = ref(false);

const {
  sortField,
  sortOrder,
  fieldOptions,
  sortParams,
  toggleSortOrder,
  orderButtonLabel,
} = useLibraryListSort(`rengu-flow-library-sort-${props.kind}`, {
  kind: props.kind,
});

function onToggleBrowserSort() {
  toggleSortOrder();
  loadBrowser(1);
}

watch([sortField, sortOrder], () => {
  if (browserOpen.value) loadBrowser(1);
});

const searchPlaceholder = computed(() =>
  props.kind === "config"
    ? "Search configs (id, model, dataset)…"
    : "Search datasets (id, folder paths)…"
);

const browserTitle = computed(() =>
  props.kind === "config" ? "Configuration library" : "Dataset library"
);

watch(
  () => props.modelValue,
  (id) => {
    if (id) searchText.value = id;
  },
  { immediate: true }
);

async function searchPage(q: string, page: number, pageSize: number): Promise<{ items: LibraryRow[]; total: number }> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
    q: q || "",
    ...sortParams(),
  });
  if (props.kind === "config") {
    return api.searchConfigs(params) as Promise<{ items: ConfigSearchItem[]; total: number }>;
  }
  return api.searchDatasets(params) as Promise<{ items: DatasetSearchItem[]; total: number }>;
}

function itemHint(item: LibraryRow): string {
  if (props.kind === "config") {
    const parts: string[] = [];
    if ("model_type" in item && item.model_type) parts.push(String(item.model_type));
    if ("dataset_ref" in item && item.dataset_ref) parts.push(String(item.dataset_ref));
    return parts.join(" · ");
  }
  const n = "directory_count" in item ? item.directory_count : undefined;
  return n != null ? `${n} directories` : "";
}

async function fetchSuggestions(
  queryString: string,
  cb: (rows: Array<LibraryRow & { value: string | number; hint: string }>) => void
) {
  try {
    const data = await searchPage(queryString || "", 1, 15);
    const items = (data.items || []).map((row) => ({
      ...row,
      value: row.id,
      hint: itemHint(row),
    }));
    cb(items);
  } catch {
    cb([]);
  }
}

function onSuggestSelect(item: Record<string, unknown>) {
  const id = item?.id as string | number | undefined;
  if (id != null) {
    emit("update:modelValue", id);
    emit("open", id);
  }
}

function onSearchEnter() {
  const q = searchText.value.trim();
  browserQuery.value = q;
  openBrowser();
  loadBrowser(1);
}

function openBrowser() {
  browserOpen.value = true;
}

function onBrowserOpen() {
  browserQuery.value = searchText.value.trim();
  loadBrowser(1);
}

async function loadBrowser(page = browserPage.value) {
  browserLoading.value = true;
  browserPage.value = page;
  try {
    const data = await searchPage(browserQuery.value, page, browserPageSize.value);
    browserItems.value = data.items || [];
    browserTotal.value = data.total ?? 0;
  } catch {
    browserItems.value = [];
    browserTotal.value = 0;
  } finally {
    browserLoading.value = false;
  }
}

function onPageSizeChange(): void {
  loadBrowser(1);
}

function selectAndOpen(id: string | number) {
  emit("update:modelValue", id);
  emit("open", id);
  browserOpen.value = false;
}

function onRowClick(row: LibraryRow) {
  selectAndOpen(row.id);
}

function onAction(command: string) {
  emitAction(command, props.modelValue);
}

function emitAction(command: string, id: string | number | null) {
  if (!id && command !== "delete") return;
  if (command === "open") {
    emit("open", id);
    return;
  }
  if (command === "duplicate") {
    emit("duplicate", id);
    return;
  }
  if (command === "delete") {
    emit("delete", id);
    return;
  }
  if (command === "start-job") {
    emit("start-job", id);
    return;
  }
  if (command === "use-job") {
    emit("use-for-job", id);
    return;
  }
  if (command === "copy-new") {
    emit("new-from-copy", id);
  }
}

defineExpose({ refreshBrowser: () => loadBrowser(browserPage.value) });
</script>

<style scoped>
.library-selector {
  margin-bottom: 12px;
}
.library-selector__row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.library-selector__search {
  flex: 1;
  min-width: 200px;
  max-width: 420px;
}
.library-selector__active {
  display: block;
  margin-top: 6px;
}
.suggest-row {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  line-height: 1.4;
}
.suggest-id {
  font-weight: 500;
}
.suggest-hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.browser-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 12px;
}
.browser-toolbar .el-input {
  flex: 1;
  min-width: 200px;
}
.browser-table {
  width: 100%;
  cursor: pointer;
}
.browser-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 12px;
}
</style>

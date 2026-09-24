<template>
  <div class="tag-freq">
    <div class="tag-freq__toolbar">
      <el-input
        v-model="search"
        placeholder="Filter tags…"
        size="small"
        clearable
        :prefix-icon="Search"
      />
      <el-select :model-value="scope" size="small" class="tag-freq__scope" @change="emit('scope-change', $event)">
        <el-option label="Line 1" value="line1" />
        <el-option label="Tag lines" value="tag_lines" />
        <el-option label="All lines" value="all_lines" />
      </el-select>
    </div>
    <div class="tag-freq__prune">
      <el-input-number v-model="pruneBelow" :min="2" size="small" controls-position="right" />
      <el-button size="small" @click="emit('prune', pruneBelow)">
        Prune tags seen &lt; n times
      </el-button>
    </div>
    <div ref="tableWrapRef" class="tag-freq__table-wrap">
      <el-table-v2
        :columns="columns"
        :data="sortedTags"
        :width="tableWidth"
        :height="tableHeight"
        :row-height="32"
        :header-height="32"
        row-key="tag"
        :sort-by="sortBy"
        :row-event-handlers="rowEventHandlers"
        class="tag-freq__table"
        @column-sort="handleColumnSort"
      >
        <template #empty>
          <div class="tag-freq__empty">
            <el-empty description="No tags" :image-size="48" />
          </div>
        </template>
      </el-table-v2>
    </div>
    <el-dialog v-model="renameOpen" title="Rename tag" width="360px" append-to-body>
      <el-form label-position="top">
        <el-form-item :label="`Rename '${renameFrom}' to:`">
          <el-input v-model="renameTo" autofocus @keydown.enter.prevent="confirmRename" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="renameOpen = false">Cancel</el-button>
        <el-button type="primary" :disabled="!renameTo.trim()" @click="confirmRename">
          Rename
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, ref } from "vue";
import { ElButton, TableV2SortOrder } from "element-plus";
import type { Column, RowEventHandlers } from "element-plus";
import { Delete, EditPen, Search } from "@element-plus/icons-vue";

interface TagRow {
  tag: string;
  count: number;
}

const props = defineProps<{
  tags: TagRow[];
  scope: string;
}>();

const emit = defineEmits<{
  (e: "select-tag", tag: string): void;
  (e: "remove-tag", tag: string): void;
  (e: "rename-tag", from: string, to: string): void;
  (e: "prune", minCount: number): void;
  (e: "scope-change", scope: string): void;
}>();

const search = ref("");
const pruneBelow = ref(3);
const renameOpen = ref(false);
const renameFrom = ref("");
const renameTo = ref("");

const filteredTags = computed(() => {
  const needle = search.value.trim().toLowerCase();
  if (!needle) return props.tags;
  return props.tags.filter((t) => t.tag.toLowerCase().includes(needle));
});

// el-table-v2 only toggles asc/desc on repeated header clicks (no third
// "unsorted" click like el-table); this mirrors that 2-state behavior.
const sortState = ref<{ key: string; order: TableV2SortOrder } | null>(null);
const sortBy = computed(() => sortState.value ?? { key: "", order: TableV2SortOrder.ASC });

function handleColumnSort(params: { key: string | number | symbol }): void {
  const key = String(params.key);
  if (sortState.value?.key === key) {
    sortState.value = {
      key,
      order: sortState.value.order === TableV2SortOrder.ASC ? TableV2SortOrder.DESC : TableV2SortOrder.ASC,
    };
  } else {
    sortState.value = { key, order: TableV2SortOrder.ASC };
  }
}

const sortedTags = computed(() => {
  const state = sortState.value;
  if (!state) return filteredTags.value;
  const dir = state.order === TableV2SortOrder.ASC ? 1 : -1;
  if (state.key === "count") {
    return [...filteredTags.value].sort((a, b) => (a.count - b.count) * dir);
  }
  return [...filteredTags.value].sort((a, b) => a.tag.localeCompare(b.tag) * dir);
});

function startRename(tag: string): void {
  renameFrom.value = tag;
  renameTo.value = tag;
  renameOpen.value = true;
}

function confirmRename(): void {
  const to = renameTo.value.trim();
  if (!to || to === renameFrom.value) {
    renameOpen.value = false;
    return;
  }
  emit("rename-tag", renameFrom.value, to);
  renameOpen.value = false;
}

function actionsCellRenderer(rowData: TagRow) {
  return h("span", { class: "tag-freq__actions" }, [
    h(ElButton, {
      size: "small",
      text: true,
      icon: EditPen,
      title: "Rename tag",
      onClick: (e: MouseEvent) => {
        e.stopPropagation();
        startRename(rowData.tag);
      },
    }),
    h(ElButton, {
      size: "small",
      text: true,
      type: "danger",
      icon: Delete,
      title: "Remove tag from all images",
      onClick: (e: MouseEvent) => {
        e.stopPropagation();
        emit("remove-tag", rowData.tag);
      },
    }),
  ]);
}

const columns = computed<Column<TagRow>[]>(() => [
  {
    key: "tag",
    dataKey: "tag",
    title: "Tag",
    width: 160,
    flexGrow: 1,
    cellRenderer: ({ cellData }) => h("span", { class: "tag-freq__cell-tag", title: String(cellData) }, String(cellData)),
  },
  {
    key: "count",
    dataKey: "count",
    title: "#",
    width: 64,
    align: "right",
    sortable: true,
  },
  {
    key: "actions",
    title: "",
    width: 92,
    align: "right",
    cellRenderer: ({ rowData }) => actionsCellRenderer(rowData as TagRow),
  },
]);

const rowEventHandlers: RowEventHandlers = {
  onClick: ({ rowData }) => emit("select-tag", (rowData as TagRow).tag),
};

// el-table-v2 requires numeric width/height (no CSS height:100%). Measure the
// wrapping flex cell via ResizeObserver, falling back to a sane default when
// no real layout is available (e.g. jsdom/happy-dom in tests) so the table
// still renders a bounded window of rows instead of collapsing to zero.
const tableWrapRef = ref<HTMLElement | null>(null);
const tableWidth = ref(320);
const tableHeight = ref(420);
let resizeObserver: ResizeObserver | undefined;

onMounted(() => {
  if (typeof ResizeObserver === "undefined" || !tableWrapRef.value) return;
  resizeObserver = new ResizeObserver((entries) => {
    const entry = entries[0];
    if (!entry) return;
    const { width, height } = entry.contentRect;
    if (width > 0) tableWidth.value = width;
    if (height > 0) tableHeight.value = height;
  });
  resizeObserver.observe(tableWrapRef.value);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
});
</script>

<style scoped>
.tag-freq {
  display: flex;
  flex-direction: column;
  gap: 8px;
  height: 100%;
  min-height: 0;
}
.tag-freq__toolbar {
  display: flex;
  gap: 8px;
}
.tag-freq__scope {
  width: 110px;
  flex-shrink: 0;
}
.tag-freq__prune {
  display: flex;
  gap: 8px;
  align-items: center;
}
.tag-freq__table-wrap {
  flex: 1;
  min-height: 0;
  cursor: pointer;
}
.tag-freq__table :deep(.el-table-v2__row) {
  cursor: pointer;
}
.tag-freq__cell-tag {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tag-freq__actions {
  display: flex;
  gap: 2px;
  justify-content: flex-end;
}
.tag-freq__empty {
  padding: 16px 0;
}
</style>

<template>
  <div class="dp-table-wrap" :class="{ 'dp-table-wrap--scroll': scrollable }">
    <el-table
      :data="items"
      stripe
      class="dp-table"
      :max-height="scrollable ? tableMaxHeight : undefined"
      :row-class-name="rowClassName"
      @row-click="onRowClick"
    >
      <el-table-column v-if="showCheck" width="44" align="center" class-name="dp-table-check-col">
        <template #default="{ row }">
          <el-icon v-if="row.active" class="dp-table-check"><Check /></el-icon>
        </template>
      </el-table-column>

      <el-table-column :label="titleLabel" min-width="160" show-overflow-tooltip>
        <template #default="{ row }">
          <span class="dp-table-title">{{ row.title }}</span>
        </template>
      </el-table-column>

      <el-table-column
        :label="subtitleLabel"
        min-width="200"
        show-overflow-tooltip
        class-name="dp-table-subtitle-col"
      >
        <template #default="{ row }">
          <span class="dp-table-subtitle">{{ row.subtitle || "—" }}</span>
        </template>
      </el-table-column>

      <el-table-column v-if="$slots.tags" :label="tagsLabel" min-width="140">
        <template #default="{ row }">
          <div class="dp-table-tags">
            <slot name="tags" :item="(row as T)" />
          </div>
        </template>
      </el-table-column>

      <el-table-column
        v-if="$slots.actions"
        label=""
        :width="actionsColumnWidth"
        align="right"
        class-name="dp-table-actions-col"
      >
        <template #default="{ row }">
          <slot name="actions" :item="(row as T)" />
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup lang="ts" generic="T extends DatasetPreviewItem = DatasetPreviewItem">
import { Check } from "@element-plus/icons-vue";
import type { DatasetPreviewItem } from "./DatasetPreviewCollection.vue";

withDefaults(
  defineProps<{
    items?: T[];
    scrollable?: boolean;
    showCheck?: boolean;
    titleLabel?: string;
    subtitleLabel?: string;
    tagsLabel?: string;
    tableMaxHeight?: string | number;
    actionsColumnWidth?: number;
  }>(),
  {
    items: () => [],
    scrollable: false,
    showCheck: false,
    titleLabel: "Name",
    subtitleLabel: "Details",
    tagsLabel: "Info",
    tableMaxHeight: "min(68vh, 680px)",
    actionsColumnWidth: 112,
  }
);

const emit = defineEmits<{ "item-click": [item: T] }>();

defineSlots<{
  tags?(props: { item: T }): unknown;
  actions?(props: { item: T }): unknown;
}>();

function rowClassName({ row }: { row: T }) {
  const classes = ["dp-table-row"];
  if (row.warning) classes.push("dp-table-row--warning");
  if (row.active) classes.push("dp-table-row--active");
  return classes.join(" ");
}

function onRowClick(row: T) {
  emit("item-click", row);
}
</script>

<style scoped>
.dp-table-wrap {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 8px;
  overflow: hidden;
}
.dp-table-wrap--scroll {
  overflow: auto;
}
.dp-table {
  width: 100%;
  --el-table-tr-bg-color: var(--el-fill-color-blank);
}
.dp-table :deep(.el-table__row) {
  cursor: pointer;
}
.dp-table :deep(.dp-table-row--warning) {
  --el-table-tr-bg-color: var(--el-color-warning-light-9);
}
.dp-table :deep(.dp-table-row--active) {
  --el-table-tr-bg-color: var(--el-color-primary-light-9);
}
.dp-table-title {
  font-weight: 600;
  font-size: 13px;
}
.dp-table-subtitle {
  font-size: 12px;
  font-family: ui-monospace, Menlo, Monaco, Consolas, monospace;
  color: var(--el-text-color-secondary);
}
.dp-table-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.dp-table-check {
  color: var(--el-color-primary);
  font-size: 16px;
}
.dp-table :deep(.dp-table-actions-col .cell) {
  overflow: visible;
}
</style>

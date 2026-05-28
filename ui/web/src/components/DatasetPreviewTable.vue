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
            <slot name="tags" :item="row" />
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
          <slot name="actions" :item="row" />
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup lang="ts">
import { Check } from "@element-plus/icons-vue";
import type { PropType } from "vue";
import type { DatasetPreviewItem } from "./DatasetPreviewCollection.vue";

defineProps({
  items: { type: Array as PropType<DatasetPreviewItem[]>, default: () => [] },
  scrollable: { type: Boolean, default: false },
  showCheck: { type: Boolean, default: false },
  titleLabel: { type: String, default: "Name" },
  subtitleLabel: { type: String, default: "Details" },
  tagsLabel: { type: String, default: "Info" },
  tableMaxHeight: { type: [String, Number], default: "min(68vh, 680px)" },
  actionsColumnWidth: { type: Number, default: 112 },
});

const emit = defineEmits(["item-click"]);

function rowClassName({ row }: { row: DatasetPreviewItem }) {
  const classes = ["dp-table-row"];
  if (row.warning) classes.push("dp-table-row--warning");
  if (row.active) classes.push("dp-table-row--active");
  return classes.join(" ");
}

function onRowClick(row: DatasetPreviewItem) {
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

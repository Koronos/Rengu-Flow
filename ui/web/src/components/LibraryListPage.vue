<template>
  <div class="library-list-page">
    <slot name="banner" />

    <div class="page-toolbar">
      <slot name="toolbar" />
    </div>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <div v-loading="loading" class="list-body">
      <el-empty
        v-if="!loading && !items.length"
        :description="emptyDescription"
        :image-size="64"
      >
        <slot name="empty-action" />
      </el-empty>
      <DatasetPreviewCollection
        v-else
        :items="items"
        :view-mode="viewMode"
        :show-check="showCheck"
        :table-subtitle-label="tableSubtitleLabel"
        :table-actions-column-width="tableActionsColumnWidth"
        @item-click="$emit('item-click', $event)"
      >
        <template #actions="slotProps">
          <slot name="actions" v-bind="slotProps" />
        </template>
      </DatasetPreviewCollection>
    </div>

    <slot name="footer" />
  </div>
</template>

<script setup lang="ts" generic="T extends DatasetPreviewItem = DatasetPreviewItem">
import { ElLoadingDirective } from "element-plus";
import DatasetPreviewCollection, {
  type DatasetPreviewItem,
} from "./DatasetPreviewCollection.vue";

withDefaults(
  defineProps<{
    loading?: boolean;
    error?: string;
    items?: T[];
    viewMode?: string;
    showCheck?: boolean;
    emptyDescription?: string;
    tableSubtitleLabel?: string;
    tableActionsColumnWidth?: number;
  }>(),
  {
    loading: false,
    error: "",
    items: () => [],
    viewMode: "cards",
    showCheck: false,
    emptyDescription: "No items yet",
    tableSubtitleLabel: "Summary",
    tableActionsColumnWidth: 112,
  }
);

defineEmits<{ "item-click": [item: T] }>();

defineSlots<{
  banner?(): unknown;
  toolbar?(): unknown;
  "empty-action"?(): unknown;
  actions?(props: { item: T }): unknown;
  footer?(): unknown;
}>();

const vLoading = ElLoadingDirective;
</script>

<style scoped>
.list-body {
  min-height: 120px;
}
</style>

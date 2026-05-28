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

<script setup lang="ts">
import { ElLoadingDirective } from "element-plus";
import DatasetPreviewCollection, {
  type DatasetPreviewItem,
} from "./DatasetPreviewCollection.vue";
import type { PropType } from "vue";

defineProps({
  loading: { type: Boolean, default: false },
  error: { type: String, default: "" },
  items: { type: Array as PropType<DatasetPreviewItem[]>, default: () => [] },
  viewMode: { type: String, default: "cards" },
  showCheck: { type: Boolean, default: false },
  emptyDescription: { type: String, default: "No items yet" },
  tableSubtitleLabel: { type: String, default: "Summary" },
  tableActionsColumnWidth: { type: Number, default: 112 },
});

defineEmits<{
  "item-click": [item: DatasetPreviewItem];
}>();

const vLoading = ElLoadingDirective;
</script>

<style scoped>
.list-body {
  min-height: 120px;
}
</style>

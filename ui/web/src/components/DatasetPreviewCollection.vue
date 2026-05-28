<template>
  <div class="dp-collection">
    <DatasetPreviewGrid v-if="viewMode === 'cards'" :scrollable="scrollable" :dense="dense">
      <DatasetPreviewCard
        v-for="item in items"
        :key="item.key"
        :title="item.title"
        :subtitle="item.subtitle"
        :thumbs="item.thumbs"
        :thumb-source="item.thumbSource"
        :active="item.active"
        :show-check="showCheck"
        :warning="item.warning"
        :fallback-text="item.fallbackText || '…'"
        :stacked="item.stacked"
        @click="$emit('item-click', item)"
      >
        <template v-if="$slots.tags" #tags>
          <slot name="tags" :item="item" />
        </template>
        <template v-if="$slots.actions" #actions>
          <slot name="actions" :item="item" />
        </template>
      </DatasetPreviewCard>
    </DatasetPreviewGrid>

    <DatasetPreviewList v-else-if="viewMode === 'list'" :scrollable="scrollable">
      <DatasetPreviewRow
        v-for="item in items"
        :key="item.key"
        :title="item.title"
        :subtitle="item.subtitle"
        :thumbs="item.thumbs"
        :thumb-source="item.thumbSource"
        :warning="item.warning"
        :active="item.active"
        :fallback-text="item.fallbackText || '…'"
        @click="$emit('item-click', item)"
      >
        <template v-if="$slots.tags" #tags>
          <slot name="tags" :item="item" />
        </template>
        <template v-if="$slots.actions" #actions>
          <slot name="actions" :item="item" />
        </template>
      </DatasetPreviewRow>
    </DatasetPreviewList>

    <DatasetPreviewTable
      v-else
      :items="items"
      :scrollable="scrollable"
      :show-check="showCheck"
      :title-label="tableTitleLabel"
      :subtitle-label="tableSubtitleLabel"
      :actions-column-width="tableActionsColumnWidth"
      @item-click="$emit('item-click', $event)"
    >
      <template v-if="$slots.actions" #actions="slotProps">
        <slot name="actions" v-bind="slotProps" />
      </template>
    </DatasetPreviewTable>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from "vue";
import DatasetPreviewCard from "./DatasetPreviewCard.vue";
import DatasetPreviewGrid from "./DatasetPreviewGrid.vue";
import DatasetPreviewList from "./DatasetPreviewList.vue";
import DatasetPreviewRow from "./DatasetPreviewRow.vue";
import DatasetPreviewTable from "./DatasetPreviewTable.vue";
import type { ThumbSource } from "../lib/previewThumbs";
import type { DirectoryFormRow } from "../lib/datasetDirectoryForm";

export interface DatasetPreviewItem {
  key: string;
  id?: string | number;
  title?: string;
  subtitle?: string;
  thumbSource?: ThumbSource | null;
  thumbs?: string[];
  fallbackText?: string;
  warning?: boolean;
  active?: boolean;
  stacked?: boolean;
  /** Directory tab rows (edit dialog, tags, folder stats). */
  dir?: DirectoryFormRow;
  index?: number;
  overrideCount?: number;
}

defineProps({
  items: { type: Array as PropType<DatasetPreviewItem[]>, default: () => [] },
  viewMode: { type: String, default: "cards" },
  scrollable: { type: Boolean, default: false },
  dense: { type: Boolean, default: false },
  showCheck: { type: Boolean, default: false },
  tableTitleLabel: { type: String, default: "Name" },
  tableSubtitleLabel: { type: String, default: "Details" },
  tableActionsColumnWidth: { type: Number, default: 112 },
});

defineEmits(["item-click"]);
</script>

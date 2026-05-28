import { computed, ref, type Ref } from "vue";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";

export function useLibraryListSelection(items: Ref<DatasetPreviewItem[]>) {
  const selectedId = ref<string | number | null>(null);

  const previewItems = computed((): DatasetPreviewItem[] =>
    items.value.map((item) => ({
      ...item,
      active: item.id != null && String(item.id) === String(selectedId.value),
    }))
  );

  function selectItem(item: DatasetPreviewItem): void {
    if (item?.id == null) return;
    selectedId.value = item.id;
  }

  function clearSelection(): void {
    selectedId.value = null;
  }

  return { selectedId, previewItems, selectItem, clearSelection };
}

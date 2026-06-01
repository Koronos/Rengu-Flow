import { computed, ref, watch, type Ref } from "vue";
import { api } from "../api";
import {
  canonicalDatasetRef,
  libraryDatasetIdFromRef,
} from "../lib/datasetLibraryRef";
import { libraryThumbSource, pathThumbSource } from "../lib/previewThumbs";
import { useResolvedDatasetLabels } from "./useResolvedDatasetLabels";
import type { DatasetPreviewItem } from "../components/DatasetPreviewCollection.vue";

/** Selected-dataset preview item: keeps the original entry string + library id. */
export interface SelectedDatasetPreviewItem extends DatasetPreviewItem {
  path: string;
  libraryId: string | null;
}

/**
 * Build rich {@link DatasetPreviewItem}s for the training `dataset` form value,
 * reusing the same preview components as the Datasets library section.
 *
 * Library refs get a thumbnail + folder-count subtitle (fetched lazily); raw
 * paths fall back to a path-based thumbnail.
 */
export function useSelectedDatasetPreviews(entries: Ref<string[]>) {
  const { labelFor } = useResolvedDatasetLabels(entries);

  /** Cached folder counts by library id (avoids refetching on every render). */
  const folderCounts = ref<Record<string, number>>({});

  async function fetchCounts(ids: string[]): Promise<void> {
    await Promise.all(
      ids.map(async (id) => {
        try {
          const row = (await api.getDataset(id)) as {
            meta?: { folder_count?: number; directory_count?: number };
            directory_count?: number;
          };
          const count =
            row.meta?.folder_count ??
            row.meta?.directory_count ??
            row.directory_count;
          if (typeof count === "number") {
            folderCounts.value = { ...folderCounts.value, [id]: count };
          }
        } catch {
          /* leave subtitle without a folder count */
        }
      })
    );
  }

  watch(
    entries,
    (list) => {
      const ids = [
        ...new Set(
          list
            .map((entry) => libraryDatasetIdFromRef(entry))
            .filter((id): id is string => !!id && !(id in folderCounts.value))
        ),
      ];
      if (ids.length) void fetchCounts(ids);
    },
    { immediate: true }
  );

  /** Re-fetch folder counts for the current entries (e.g. after editing a dataset). */
  async function refresh(): Promise<void> {
    const ids = [
      ...new Set(
        entries.value
          .map((entry) => libraryDatasetIdFromRef(entry))
          .filter((id): id is string => !!id)
      ),
    ];
    if (!ids.length) return;
    const next = { ...folderCounts.value };
    ids.forEach((id) => delete next[id]);
    folderCounts.value = next;
    await fetchCounts(ids);
  }

  function subtitleFor(entry: string, libraryId: string | null): string {
    if (!libraryId) return entry;
    const count = folderCounts.value[libraryId];
    if (typeof count !== "number") return `Library #${libraryId}`;
    return `Library #${libraryId} · ${count} ${count === 1 ? "folder" : "folders"}`;
  }

  const previewItems = computed((): SelectedDatasetPreviewItem[] =>
    entries.value.map((entry) => {
      const libraryId = libraryDatasetIdFromRef(entry);
      return {
        key: canonicalDatasetRef(entry),
        id: libraryId ?? undefined,
        path: entry,
        libraryId,
        title: labelFor(entry),
        subtitle: subtitleFor(entry, libraryId),
        thumbSource: libraryId
          ? libraryThumbSource(libraryId)
          : pathThumbSource(entry),
        fallbackText: "DS",
      };
    })
  );

  return { previewItems, refresh };
}

import { ref, type Ref } from "vue";
import { formatError } from "../lib/formatError";
import { useLatestAsync } from "./useLatestAsync";
import type { Paginated } from "../types/api";

export type LibrarySearchParams = {
  q: string;
  page: number;
  page_size: number;
  sort: string;
  order: string;
};

/** Shared debounced search + loading state for dataset/config library list pages. */
export function useDebouncedLibrarySearch<T>(
  searchFn: (params: LibrarySearchParams) => Promise<Paginated<T>>,
  sortParams: () => { sort: string; order: string },
  { pageSize = 100, debounceMs = 300 } = {}
) {
  const rawItems = ref([]) as Ref<T[]>;
  const loading = ref(false);
  const error = ref("");
  const query = ref("");
  const guard = useLatestAsync();

  async function load(): Promise<void> {
    const token = guard.begin();
    loading.value = true;
    error.value = "";
    try {
      const data = await searchFn({
        q: query.value.trim(),
        page: 1,
        page_size: pageSize,
        ...sortParams(),
      });
      if (!guard.isCurrent(token)) return; // a newer search superseded this one
      rawItems.value = data.items || [];
    } catch (e) {
      if (!guard.isCurrent(token)) return;
      error.value = formatError(e);
      rawItems.value = [];
    } finally {
      if (guard.isCurrent(token)) loading.value = false;
    }
  }

  function scheduleSearch(): void {
    guard.schedule(load, debounceMs);
  }

  return { rawItems, loading, error, query, load, scheduleSearch };
}

import { ref, type Ref } from "vue";
import { formatError } from "../lib/formatError";
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
  let searchTimer: ReturnType<typeof setTimeout> | undefined;

  async function load(): Promise<void> {
    loading.value = true;
    error.value = "";
    try {
      const data = await searchFn({
        q: query.value.trim(),
        page: 1,
        page_size: pageSize,
        ...sortParams(),
      });
      rawItems.value = data.items || [];
    } catch (e) {
      error.value = formatError(e);
      rawItems.value = [];
    } finally {
      loading.value = false;
    }
  }

  function scheduleSearch(): void {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(load, debounceMs);
  }

  return { rawItems, loading, error, query, load, scheduleSearch };
}

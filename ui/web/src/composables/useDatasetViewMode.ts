import { ref, watch } from "vue";

const DEFAULT_KEY = "renga-flow-dataset-preview-view";
type DatasetViewMode = "cards" | "list" | "table";

const VALID_MODES = new Set<DatasetViewMode>(["cards", "list", "table"]);

function normalizeStoredMode(stored: string | null): DatasetViewMode {
  if (stored === "text") return "table";
  return stored && VALID_MODES.has(stored as DatasetViewMode)
    ? (stored as DatasetViewMode)
    : "cards";
}

/**
 * Persisted cards | list | table toggle shared across dataset UI sections.
 * @param {string} [storageKey]
 */
export function useDatasetViewMode(
  storageKey = DEFAULT_KEY,
  defaultMode: DatasetViewMode = "cards"
) {
  const stored =
    typeof localStorage !== "undefined" ? localStorage.getItem(storageKey) : null;
  const viewMode = ref(
    stored != null ? normalizeStoredMode(stored) : defaultMode
  );

  watch(viewMode, (mode: DatasetViewMode) => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(storageKey, mode);
    }
  });

  return { viewMode };
}

export const DATASET_LIBRARY_VIEW_KEY = "renga-flow-dataset-library-view";
export const DATASET_DIRECTORY_VIEW_KEY = "renga-flow-dataset-directory-view";

import { ref, watch } from "vue";

const DEFAULT_KEY = "renga-flow-dataset-preview-view";

const VALID_MODES = new Set(["cards", "list", "table"]);

function normalizeStoredMode(stored) {
  if (stored === "text") return "table";
  return VALID_MODES.has(stored) ? stored : "cards";
}

/**
 * Persisted cards | list | table toggle shared across dataset UI sections.
 * @param {string} [storageKey]
 */
export function useDatasetViewMode(storageKey = DEFAULT_KEY) {
  const stored =
    typeof localStorage !== "undefined" ? localStorage.getItem(storageKey) : null;
  const viewMode = ref(normalizeStoredMode(stored));

  watch(viewMode, (mode) => {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(storageKey, mode);
    }
  });

  return { viewMode };
}

export const DATASET_LIBRARY_VIEW_KEY = "renga-flow-dataset-library-view";
export const DATASET_DIRECTORY_VIEW_KEY = "renga-flow-dataset-directory-view";

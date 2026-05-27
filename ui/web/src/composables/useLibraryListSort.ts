import { computed, ref, watch } from "vue";

const DATASET_SORT_FIELDS = [
  { value: "id", label: "ID" },
  { value: "name", label: "Name" },
  { value: "created_at", label: "Created" },
  { value: "updated_at", label: "Updated" },
];

const CONFIG_SORT_FIELDS = [
  { value: "id", label: "ID" },
  { value: "name", label: "Run name" },
  { value: "created_at", label: "Created" },
  { value: "updated_at", label: "Updated" },
];

/**
 * Persisted library list sort (`sort` + `order` API query params).
 *
 * @param {string} storageKey localStorage key
 * @param {{ kind?: 'dataset' | 'config', defaultField?: string, defaultOrder?: 'asc'|'desc' }} options
 */
export function useLibraryListSort(
  storageKey,
  { kind = "dataset", defaultField = "id", defaultOrder = "desc" } = {}
) {
  const fieldOptions = kind === "config" ? CONFIG_SORT_FIELDS : DATASET_SORT_FIELDS;
  const allowedFields = new Set(fieldOptions.map((f) => f.value));

  const sortField = ref(defaultField);
  const sortOrder = ref(defaultOrder);

  function loadStored() {
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return;
      if (raw.includes(":")) {
        const [field, order] = raw.split(":");
        if (allowedFields.has(field)) sortField.value = field;
        if (order === "asc" || order === "desc") sortOrder.value = order;
        return;
      }
      const parsed = JSON.parse(raw);
      if (parsed && allowedFields.has(parsed.field)) sortField.value = parsed.field;
      if (parsed && (parsed.order === "asc" || parsed.order === "desc")) {
        sortOrder.value = parsed.order;
      }
    } catch {
      /* ignore */
    }
  }

  loadStored();

  watch(
    [sortField, sortOrder],
    () => {
      try {
        localStorage.setItem(
          storageKey,
          JSON.stringify({ field: sortField.value, order: sortOrder.value })
        );
      } catch {
        /* ignore */
      }
    },
    { deep: true }
  );

  function sortParams() {
    return { sort: sortField.value, order: sortOrder.value };
  }

  function toggleSortOrder() {
    sortOrder.value = sortOrder.value === "desc" ? "asc" : "desc";
  }

  const orderButtonLabel = computed(() =>
    sortOrder.value === "desc" ? "Descending" : "Ascending"
  );

  return {
    sortField,
    sortOrder,
    fieldOptions,
    sortParams,
    toggleSortOrder,
    orderButtonLabel,
  };
}

import { ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { DuplicateConfigResult } from "../types/api";

/**
 * Duplicate/delete actions for a library list view.
 *
 * Defaults to the dataset endpoints, which is what every existing caller wants. `duplicate` /
 * `remove` / `label` let another library — workflows — reuse the busy flag, the confirm prompt and
 * the toast wording instead of re-implementing all three slightly differently.
 */
export function useLibraryCrudActions(options: {
  onDeleted?: () => void;
  onDuplicated?: (id: string | number) => void;
  /** What one item is called in the confirm prompt. */
  label?: string;
  duplicate?: (id: string | number) => Promise<DuplicateConfigResult>;
  remove?: (id: string | number) => Promise<unknown>;
}) {
  const busy = ref(false);
  const label = options.label ?? "dataset";
  const duplicateItem = options.duplicate ?? ((id: string | number) => api.duplicateDataset(id));
  const removeItem = options.remove ?? ((id: string | number) => api.deleteDataset(id));

  async function duplicateSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    busy.value = true;
    try {
      const r = await duplicateItem(id);
      ElMessage.success(`Duplicated as ${r.id}`);
      options.onDuplicated?.(r.id);
    } catch (e) {
      ElMessage.error(formatError(e));
    } finally {
      busy.value = false;
    }
  }

  async function deleteSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    try {
      await ElMessageBox.confirm(`Delete ${label} "${id}"?`, "Confirm", { type: "warning" });
    } catch {
      return;
    }
    busy.value = true;
    try {
      await removeItem(id);
      ElMessage.success("Deleted");
      options.onDeleted?.();
    } catch (e) {
      ElMessage.error(formatError(e));
    } finally {
      busy.value = false;
    }
  }

  return { busy, duplicateSelected, deleteSelected };
}

import { ref } from "vue";
import type { Router } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { DuplicateConfigResult } from "../types/api";

/** Duplicate/delete actions for the dataset library list view. */
export function useLibraryCrudActions(options: { router: Router; onDeleted?: () => void }) {
  const busy = ref(false);

  async function duplicateSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    busy.value = true;
    try {
      const r = (await api.duplicateDataset(id)) as DuplicateConfigResult;
      ElMessage.success(`Duplicated as ${r.id}`);
      await options.router.push({
        name: "datasets-detail",
        params: { datasetId: String(r.id) },
      });
    } catch (e) {
      ElMessage.error(formatError(e));
    } finally {
      busy.value = false;
    }
  }

  async function deleteSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    try {
      await ElMessageBox.confirm(`Delete dataset "${id}"?`, "Confirm", { type: "warning" });
    } catch {
      return;
    }
    busy.value = true;
    try {
      await api.deleteDataset(id);
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

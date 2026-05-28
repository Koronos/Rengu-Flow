import { ref } from "vue";
import type { Router } from "vue-router";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { DuplicateConfigResult } from "../types/api";

type LibraryKind = "dataset" | "config";

export function useLibraryCrudActions(
  kind: LibraryKind,
  options: { router: Router; onDeleted?: () => void }
) {
  const busy = ref(false);

  async function duplicateSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    busy.value = true;
    try {
      const r =
        kind === "dataset"
          ? ((await api.duplicateDataset(id)) as DuplicateConfigResult)
          : ((await api.duplicate(id)) as DuplicateConfigResult);
      ElMessage.success(`Duplicated as ${r.id}`);
      if (kind === "dataset") {
        await options.router.push({
          name: "datasets-detail",
          params: { datasetId: String(r.id) },
        });
      } else {
        await options.router.push({
          name: "configs-detail",
          params: { configId: String(r.id) },
        });
      }
    } catch (e) {
      ElMessage.error(formatError(e));
    } finally {
      busy.value = false;
    }
  }

  async function deleteSelected(id: string | number | null): Promise<void> {
    if (id == null || busy.value) return;
    const label = kind === "dataset" ? "dataset" : "config";
    try {
      await ElMessageBox.confirm(`Delete ${label} "${id}"?`, "Confirm", { type: "warning" });
    } catch {
      return;
    }
    busy.value = true;
    try {
      if (kind === "dataset") await api.deleteDataset(id);
      else await api.deleteConfig(id);
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

import { ref, shallowRef } from "vue";
import { defineStore } from "pinia";

export interface DatasetSavedPayload {
  id: string;
  name: string;
  /** Library ref (`rengu-flow-dataset:<id>`) for use in a training config. */
  ref: string;
}
type OnSaved = (payload: DatasetSavedPayload) => void;

/** Shared state for the app-wide DatasetFormModalHost (list / picker / run form). */
export const useDatasetFormModalStore = defineStore("datasetFormModal", () => {
  const visible = ref(false);
  const mode = ref<"create" | "edit">("create");
  const editId = ref<string | null>(null);
  const onSavedCb = shallowRef<OnSaved | null>(null);
  const initialToml = ref<string | null>(null);

  function openCreate(opts: { onSaved?: OnSaved; initialToml?: string } = {}) {
    mode.value = "create";
    editId.value = null;
    onSavedCb.value = opts.onSaved ?? null;
    initialToml.value = opts.initialToml ?? null;
    visible.value = true;
  }

  function openEdit(id: string | number, opts: { onSaved?: OnSaved } = {}) {
    mode.value = "edit";
    editId.value = String(id);
    onSavedCb.value = opts.onSaved ?? null;
    initialToml.value = null;
    visible.value = true;
  }

  function close() {
    visible.value = false;
  }

  function notifySaved(payload: DatasetSavedPayload) {
    onSavedCb.value?.(payload);
  }

  return { visible, mode, editId, initialToml, openCreate, openEdit, close, notifySaved };
});

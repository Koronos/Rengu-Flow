import { ref, shallowRef } from "vue";

export interface DatasetSavedPayload {
  id: string;
  name: string;
  /** Library ref (`rengu-flow-dataset:<id>`) for use in a training config. */
  ref: string;
}
type OnSaved = (payload: DatasetSavedPayload) => void;

/** Shared state for the app-wide DatasetFormModalHost (list / picker / run form). */
const visible = ref(false);
const mode = ref<"create" | "edit">("create");
const editId = ref<string | null>(null);
const onSavedCb = shallowRef<OnSaved | null>(null);

export function useDatasetFormModal() {
  function openCreate(opts: { onSaved?: OnSaved } = {}) {
    mode.value = "create";
    editId.value = null;
    onSavedCb.value = opts.onSaved ?? null;
    visible.value = true;
  }

  function openEdit(id: string | number, opts: { onSaved?: OnSaved } = {}) {
    mode.value = "edit";
    editId.value = String(id);
    onSavedCb.value = opts.onSaved ?? null;
    visible.value = true;
  }

  function close() {
    visible.value = false;
  }

  function notifySaved(payload: DatasetSavedPayload) {
    onSavedCb.value?.(payload);
  }

  return { visible, mode, editId, openCreate, openEdit, close, notifySaved };
}

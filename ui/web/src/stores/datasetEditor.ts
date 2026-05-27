import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { ElMessage } from "element-plus";
import type { RouteLocationNormalizedLoaded, Router } from "vue-router";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { sanitizeDatasetForm } from "../lib/datasetFormPayload";
import { useTomlFormSync } from "../composables/useTomlFormSync";
import type { FormValues } from "../types/forms";

export const DEFAULT_DATASET_TOML = `resolutions = [1024]
frame_buckets = [1]
`;

export const useDatasetEditorStore = defineStore("datasetEditor", () => {
  const datasetId = ref<string | null>(null);
  const isNew = ref(true);
  const name = ref("New dataset");

  const content = ref(DEFAULT_DATASET_TOML);
  const form = shallowRef<FormValues | null>(null);
  const uiNotes = ref<unknown[]>([]);
  const schema = ref<Record<string, unknown> | null>(null);

  const loading = ref(false);
  const saving = ref(false);
  const syncing = ref(false);
  const error = ref("");
  const message = ref("");
  const parseError = ref("");
  const formVersion = ref(0);

  const title = computed(() => {
    const label = (name.value || "").trim();
    if (label) return label;
    if (isNew.value) return "New dataset";
    return datasetId.value ? `Dataset #${datasetId.value}` : "Dataset";
  });

  function setName(nextName: string) {
    name.value = nextName;
  }

  const tomlSync = useTomlFormSync<FormValues>({
    content,
    form,
    syncing,
    parseError,
    sanitize: sanitizeDatasetForm,
    formatError,
    onFormVersionBump: () => {
      formVersion.value += 1;
    },
    onParsed: (extras) => {
      uiNotes.value = (extras.ui_notes as unknown[]) || [];
    },
    parseToml: async (toml) => {
      const r = (await api.parseDatasetToml(toml)) as {
        ok?: boolean;
        form?: FormValues;
        error?: unknown;
        ui_notes?: unknown[];
      };
      return {
        ok: !!r.ok,
        form: r.form,
        error: r.error,
        extras: { ui_notes: r.ui_notes },
      };
    },
    renderToml: async (payload) => {
      const r = (await api.renderDatasetToml(payload)) as {
        ok?: boolean;
        content?: string;
        error?: unknown;
      };
      return {
        ok: !!r.ok,
        content: r.content,
        error: r.error,
      };
    },
  });

  async function fetchSchema() {
    schema.value = (await api.getDatasetSchema()) as Record<string, unknown>;
    return schema.value;
  }

  function patchFormField(path: string, value: unknown) {
    if (!form.value || !path) return;
    tomlSync.setForm({ ...form.value, [path]: value });
  }

  function patchDirectories(nextDirs: unknown) {
    if (!form.value) return;
    tomlSync.setForm({ ...form.value, _directories: nextDirs });
  }

  async function openFromRoute(route: RouteLocationNormalizedLoaded) {
    error.value = "";
    message.value = "";
    isNew.value = route.name === "datasets-new";
    datasetId.value = isNew.value ? null : String(route.params.datasetId || "");

    loading.value = true;
    try {
      await fetchSchema();
      if (isNew.value) {
        name.value = "New dataset";
        await tomlSync.applyToml(DEFAULT_DATASET_TOML);
        return;
      }
      if (!datasetId.value) return;
      const data = (await api.getDataset(datasetId.value)) as {
        name?: string;
        content: string;
      };
      name.value = data.name || `Dataset ${datasetId.value}`;
      await tomlSync.applyToml(data.content);
    } catch (e) {
      error.value = formatError(e);
      throw e;
    } finally {
      loading.value = false;
    }
  }

  async function save(router: Router) {
    saving.value = true;
    error.value = "";
    message.value = "";
    try {
      await tomlSync.flushSync();
      const toml = content.value;
      const payload = { content: toml, name: (name.value || "").trim() || undefined };
      if (isNew.value) {
        const r = (await api.createDataset(payload)) as { id: string | number; name?: string };
        datasetId.value = String(r.id);
        isNew.value = false;
        if (r.name) name.value = r.name;
        await tomlSync.applyToml(toml);
        ElMessage.success("Created");
        await router.replace({
          name: "datasets-detail",
          params: { datasetId: datasetId.value },
        });
      } else {
        const r = (await api.saveDataset(datasetId.value!, payload)) as { name?: string };
        if (r?.name) name.value = r.name;
        await tomlSync.applyToml(toml);
        ElMessage.success("Saved");
        message.value = "Saved.";
      }
    } catch (e) {
      error.value = formatError(e);
      ElMessage.error(error.value);
      throw e;
    } finally {
      saving.value = false;
    }
  }

  async function validate() {
    error.value = "";
    message.value = "";
    try {
      await tomlSync.flushSync();
      const r = (await api.validateDataset(content.value)) as {
        ok?: boolean;
        preview?: { total_images?: number };
        error?: unknown;
      };
      if (r.ok) {
        const n = r.preview?.total_images ?? "?";
        message.value = `Valid — ${n} images`;
        ElMessage.success(message.value);
      } else {
        error.value = formatError({ detail: r.error }) || "Invalid";
        ElMessage.error(error.value);
      }
    } catch (e) {
      error.value = formatError(e);
      ElMessage.error(error.value);
      throw e;
    }
  }

  function dispose() {
    tomlSync.resetSyncState();
    datasetId.value = null;
    isNew.value = true;
    name.value = "New dataset";
    content.value = DEFAULT_DATASET_TOML;
    form.value = null;
    uiNotes.value = [];
    schema.value = null;
    loading.value = false;
    saving.value = false;
    syncing.value = false;
    error.value = "";
    message.value = "";
    parseError.value = "";
    formVersion.value = 0;
  }

  return {
    datasetId,
    isNew,
    name,
    title,
    setName,
    content,
    form,
    uiNotes,
    schema,
    loading,
    saving,
    syncing,
    error,
    message,
    parseError,
    formVersion,
    setContent: tomlSync.setContent,
    setForm: tomlSync.setForm,
    patchFormField,
    patchDirectories,
    applyToml: tomlSync.applyToml,
    flushSync: tomlSync.flushSync,
    openFromRoute,
    save,
    validate,
    dispose,
  };
});

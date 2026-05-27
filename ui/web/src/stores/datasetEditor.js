import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { sanitizeDatasetForm } from "../lib/datasetFormPayload";

export const DEFAULT_DATASET_TOML = `resolutions = [1024]
frame_buckets = [1]
`;

const PARSE_DEBOUNCE_MS = 280;
const RENDER_DEBOUNCE_MS = 280;

export const useDatasetEditorStore = defineStore("datasetEditor", () => {
  const datasetId = ref(null);
  const isNew = ref(true);
  const name = ref("New dataset");

  const content = ref(DEFAULT_DATASET_TOML);
  const form = shallowRef(null);
  const uiNotes = ref([]);
  const schema = ref(null);

  const loading = ref(false);
  const saving = ref(false);
  const syncing = ref(false);
  const error = ref("");
  const message = ref("");
  const parseError = ref("");
  /** Bumped on every TOML→form parse so the form pane re-renders. */
  const formVersion = ref(0);

  const title = computed(() => {
    const label = (name.value || "").trim();
    if (label) return label;
    if (isNew.value) return "New dataset";
    return datasetId.value ? `Dataset #${datasetId.value}` : "Dataset";
  });

  function setName(nextName) {
    name.value = nextName;
  }

  let parseTimer = null;
  let renderTimer = null;
  /** @type {"toml-to-form" | "form-to-toml" | null} */
  let syncLock = null;
  /** @type {"toml" | "form"} */
  let lastEditSource = "toml";

  function clearSyncTimers() {
    clearTimeout(parseTimer);
    clearTimeout(renderTimer);
  }

  async function fetchSchema() {
    schema.value = await api.getDatasetSchema();
    return schema.value;
  }

  async function parseFromToml() {
    syncing.value = true;
    parseError.value = "";
    try {
      const r = await api.parseDatasetToml(content.value);
      if (!r.ok) {
        parseError.value =
          formatError({ detail: r.error }) || "Could not parse TOML for the form";
        return;
      }
      syncLock = "toml-to-form";
      const clean = sanitizeDatasetForm(r.form);
      form.value = clean;
      formVersion.value += 1;
      uiNotes.value = r.ui_notes || [];
    } catch (e) {
      parseError.value = formatError(e);
    } finally {
      syncLock = null;
      syncing.value = false;
    }
  }

  async function renderFromForm() {
    if (!form.value) return;
    syncing.value = true;
    parseError.value = "";
    try {
      const payload = sanitizeDatasetForm(form.value);
      if (!payload) {
        parseError.value = "Could not sync form to TOML (invalid form state).";
        return;
      }
      const r = await api.renderDatasetToml(payload);
      if (!r.ok) {
        parseError.value =
          formatError({ detail: r.error }) || "Could not render TOML from form";
        return;
      }
      const nextToml = r.content ?? "";
      if (nextToml === content.value) return;
      syncLock = "form-to-toml";
      content.value = nextToml;
    } catch (e) {
      parseError.value = formatError(e);
    } finally {
      syncLock = null;
      syncing.value = false;
    }
  }

  function scheduleParseFromToml() {
    if (syncLock === "form-to-toml") return;
    clearTimeout(parseTimer);
    parseTimer = setTimeout(() => parseFromToml(), PARSE_DEBOUNCE_MS);
  }

  function scheduleRenderFromForm() {
    if (syncLock === "toml-to-form") return;
    clearTimeout(renderTimer);
    renderTimer = setTimeout(() => renderFromForm(), RENDER_DEBOUNCE_MS);
  }

  function setContent(toml) {
    lastEditSource = "toml";
    clearTimeout(renderTimer);
    renderTimer = null;
    content.value = toml;
    scheduleParseFromToml();
  }

  function setForm(nextForm) {
    lastEditSource = "form";
    clearTimeout(parseTimer);
    parseTimer = null;
    const clean = sanitizeDatasetForm(nextForm);
    if (!clean) return;
    form.value = clean;
    scheduleRenderFromForm();
  }

  function patchFormField(path, value) {
    if (!form.value || !path) return;
    setForm({ ...form.value, [path]: value });
  }

  function patchDirectories(nextDirs) {
    if (!form.value) return;
    setForm({ ...form.value, _directories: nextDirs });
  }

  async function applyToml(toml) {
    clearSyncTimers();
    syncLock = null;
    lastEditSource = "toml";
    parseError.value = "";
    content.value = toml;
    await parseFromToml();
  }

  async function flushSync() {
    clearSyncTimers();
    if (lastEditSource === "form") {
      await renderFromForm();
      await parseFromToml();
    } else {
      await parseFromToml();
      await renderFromForm();
    }
  }

  async function openFromRoute(route) {
    error.value = "";
    message.value = "";
    isNew.value = route.name === "datasets-new";
    datasetId.value = isNew.value ? null : String(route.params.datasetId || "");

    loading.value = true;
    try {
      await fetchSchema();
      if (isNew.value) {
        name.value = "New dataset";
        await applyToml(DEFAULT_DATASET_TOML);
        return;
      }
      if (!datasetId.value) return;
      const data = await api.getDataset(datasetId.value);
      name.value = data.name || `Dataset ${datasetId.value}`;
      await applyToml(data.content);
    } catch (e) {
      error.value = formatError(e);
      throw e;
    } finally {
      loading.value = false;
    }
  }

  async function save(router) {
    saving.value = true;
    error.value = "";
    message.value = "";
    try {
      await flushSync();
      const toml = content.value;
      const payload = { content: toml, name: (name.value || "").trim() || undefined };
      if (isNew.value) {
        const r = await api.createDataset(payload);
        datasetId.value = String(r.id);
        isNew.value = false;
        if (r.name) name.value = r.name;
        await applyToml(toml);
        ElMessage.success("Created");
        await router.replace({
          name: "datasets-detail",
          params: { datasetId: datasetId.value },
        });
      } else {
        const r = await api.saveDataset(datasetId.value, payload);
        if (r?.name) name.value = r.name;
        await applyToml(toml);
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
      await flushSync();
      const r = await api.validateDataset(content.value);
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
    clearSyncTimers();
    syncLock = null;
    lastEditSource = "toml";
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
    setContent,
    setForm,
    patchFormField,
    patchDirectories,
    applyToml,
    flushSync,
    openFromRoute,
    save,
    validate,
    dispose,
  };
});

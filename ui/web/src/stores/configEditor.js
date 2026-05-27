import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { sanitizeConfigForm } from "../lib/configFormPayload";
import {
  getModelCapability,
  modelSupportsAdapters,
  pruneFormForModel,
} from "../lib/formUtils";

export const DEFAULT_CONFIG_TOML = `dataset = "examples/minimal_dataset.toml"

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = "path/to/sdxl.safetensors"

[adapter]
type = "lora"
rank = 16

[optimizer]
type = "adamw"
lr = 1.0e-4

lr_scheduler = "cosine"
[lr_scheduler_args]
lr_min = 0.0

epochs = 1
gradient_accumulation_steps = 1
micro_batch_size_per_gpu = 1
synthetic_num_batches = 50
logging_steps = 1
save_every_n_epochs = 1
output_dir = "output"
`;

const PARSE_DEBOUNCE_MS = 280;
const RENDER_DEBOUNCE_MS = 280;

export const useConfigEditorStore = defineStore("configEditor", () => {
  const configId = ref(null);
  const isNew = ref(false);
  const selectedMeta = ref("");
  const content = ref(DEFAULT_CONFIG_TOML);
  const form = shallowRef({ _has_adapter: true });
  const schema = ref(null);

  const loading = ref(false);
  const saving = ref(false);
  const syncing = ref(false);
  const error = ref("");
  const message = ref("");
  const parseError = ref("");
  const validationErrors = ref([]);
  const formVersion = ref(0);
  const editorTab = ref("form");

  const continuation = ref(null);
  const continuationSaveToLibrary = ref(false);
  const continuationLibraryId = ref("");

  const modelCapabilities = computed(
    () => schema.value?.registries?.model_capabilities ?? {}
  );

  const editingTitle = computed(() => {
    if (isNew.value && !configId.value) return "New config";
    if (configId.value) return `Config #${configId.value}`;
    return "Config";
  });

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

  function cleanForm(raw) {
    return sanitizeConfigForm(raw, modelCapabilities.value);
  }

  function applyModelCapabilityDefaults(target) {
    const cap = getModelCapability(modelCapabilities.value, target["model.type"]);
    if (!cap) return target;

    const next = { ...target };
    let changed = false;

    if (!modelSupportsAdapters(cap)) {
      if (next._has_adapter !== false) {
        next._has_adapter = false;
        changed = true;
      }
    } else if (!cap.full_finetune && !next._has_adapter) {
      next._has_adapter = true;
      changed = true;
    }

    if (next._has_adapter && cap.adapters?.length) {
      const allowed = cap.adapters;
      const current = next["adapter.type"];
      if (!current || !allowed.includes(current)) {
        next["adapter.type"] = allowed[0];
        changed = true;
      }
    }

    return changed ? next : target;
  }

  async function fetchSchema() {
    schema.value = await api.getSchema();
    return schema.value;
  }

  async function parseFromToml() {
    if (!(content.value || "").trim()) return;
    syncing.value = true;
    parseError.value = "";
    try {
      const r = await api.parseToml(content.value);
      if (!r.ok) {
        parseError.value =
          formatError({ detail: r.error }) || "Could not parse TOML for the form";
        return;
      }
      syncLock = "toml-to-form";
      let next = cleanForm(r.form) || { _has_adapter: true };
      next = applyModelCapabilityDefaults(next);
      form.value = next;
      formVersion.value += 1;
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
      const payload = cleanForm(form.value);
      if (!payload) {
        parseError.value = "Could not sync form to TOML (invalid form state).";
        return;
      }
      const r = await api.renderToml(payload);
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
    const clean = cleanForm(nextForm);
    if (!clean) return;
    form.value = clean;
    scheduleRenderFromForm();
  }

  function patchFormField(path, value) {
    if (!form.value || !path) return;
    let next = { ...form.value, [path]: value };
    if (path === "model.type") {
      next = pruneFormForModel(next, modelCapabilities.value);
      next = applyModelCapabilityDefaults(next);
    }
    setForm(next);
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

  function resetEditorState() {
    validationErrors.value = [];
    error.value = "";
    message.value = "";
    parseError.value = "";
  }

  async function newConfig() {
    configId.value = null;
    isNew.value = true;
    selectedMeta.value = "";
    resetEditorState();
    editorTab.value = "form";
    await applyToml(DEFAULT_CONFIG_TOML);
  }

  async function openConfig(id) {
    resetEditorState();
    const data = await api.getConfig(id);
    configId.value = id;
    isNew.value = false;
    content.value = data.content;
    editorTab.value = "form";
    try {
      const summary = await api.searchConfigs({ q: id, page: 1, page_size: 1 });
      const row = summary.items?.find((c) => c.id === id);
      if (row) {
        const parts = [];
        if (row.model_type) parts.push(row.model_type);
        if (row.dataset_ref) parts.push(row.dataset_ref);
        selectedMeta.value = parts.join(" · ");
      } else {
        selectedMeta.value = "";
      }
    } catch {
      selectedMeta.value = "";
    }
    await applyToml(data.content);
  }

  async function loadContinuation(runPath) {
    resetEditorState();
    const data = await api.getRunConfig(runPath);
    continuation.value = {
      run_dir: data.run_dir,
      resume_from: data.resume_from,
    };
    continuationLibraryId.value = `${data.run_dir.split("/").pop()}_continued`;
    configId.value = null;
    isNew.value = true;
    selectedMeta.value = "from run folder";
    editorTab.value = "form";
    await applyToml(data.content);
  }

  function clearContinuation() {
    continuation.value = null;
  }

  async function saveExisting() {
    if (!configId.value) return false;
    saving.value = true;
    error.value = "";
    message.value = "";
    try {
      await flushSync();
      await api.saveConfig(configId.value, content.value);
      message.value = "Saved.";
      ElMessage.success("Saved");
      return true;
    } catch (e) {
      error.value = formatError(e);
      ElMessage.error(error.value);
      throw e;
    } finally {
      saving.value = false;
    }
  }

  async function createNew(id) {
    const trimmed = (id || "").trim();
    if (!trimmed) return false;
    saving.value = true;
    error.value = "";
    message.value = "";
    try {
      await flushSync();
      await api.createConfig(trimmed, content.value);
      configId.value = trimmed;
      isNew.value = false;
      message.value = "Saved.";
      ElMessage.success("Created");
      return true;
    } catch (e) {
      error.value = formatError(e);
      ElMessage.error(error.value);
      throw e;
    } finally {
      saving.value = false;
    }
  }

  /** @returns {Promise<{ ok: boolean, resolution?: object, errors?: string[] }>} */
  async function validateConfig({ quiet = false } = {}) {
    error.value = "";
    validationErrors.value = [];
    if (!quiet) message.value = "";
    try {
      await flushSync();
      const r = await api.validate(content.value);
      if (r.ok) {
        const res = r.resolution || {};
        const parts = [];
        if (res.optimizer?.available) {
          parts.push(`optimizer → ${res.optimizer.resolved_class || res.optimizer.name}`);
        }
        if (res.scheduler?.available) {
          parts.push(
            `scheduler → ${res.scheduler.resolved || res.scheduler.resolved_class || res.scheduler.name}`
          );
        }
        if (!quiet) {
          message.value = parts.length ? `Valid (${parts.join("; ")})` : "Valid.";
          ElMessage.success(message.value);
        }
        return { ok: true, resolution: res };
      }
      validationErrors.value =
        Array.isArray(r.errors) && r.errors.length
          ? r.errors
          : [r.error || "Invalid configuration."];
      if (!quiet) {
        ElMessage.error(
          validationErrors.value.length === 1
            ? validationErrors.value[0]
            : `${validationErrors.value.length} issues — see list above`
        );
      }
      return { ok: false, errors: validationErrors.value };
    } catch (e) {
      error.value = formatError(e);
      if (!quiet) ElMessage.error(error.value);
      throw e;
    }
  }

  async function queueContinuation({ startNow, saveToLibrary, libraryId }) {
    if (!continuation.value) return null;
    error.value = "";
    try {
      await flushSync();
      const job = await api.continueRun({
        run_path: continuation.value.run_dir,
        content: content.value,
        save_to_library: saveToLibrary,
        config_id: saveToLibrary ? libraryId?.trim() || undefined : undefined,
        enqueue: !startNow,
        start_immediately: startNow,
      });
      ElMessage.success(startNow ? "Continuation started" : "Continuation queued");
      clearContinuation();
      return job;
    } catch (e) {
      error.value = formatError(e);
      ElMessage.error(error.value);
      throw e;
    }
  }

  async function openFromRoute(route) {
    error.value = "";
    message.value = "";
    loading.value = true;
    try {
      await fetchSchema();
      const cr = route.query.continue_run;
      if (typeof cr === "string" && cr) {
        await loadContinuation(cr);
        return;
      }
      if (route.name === "configs-new") {
        await newConfig();
        return;
      }
      const id = String(route.params.configId || "").trim();
      if (!id) throw new Error("Missing config id");
      await openConfig(id);
    } catch (e) {
      error.value = formatError(e);
      throw e;
    } finally {
      loading.value = false;
    }
  }

  async function bootstrapPickForJob(storedConfigId) {
    if (!storedConfigId) return;
    loading.value = true;
    try {
      if (!schema.value) await fetchSchema();
      await openConfig(storedConfigId);
    } catch {
      /* config may have been deleted */
    } finally {
      loading.value = false;
    }
  }

  function clearSelection() {
    configId.value = null;
    isNew.value = false;
    selectedMeta.value = "";
    content.value = "";
    form.value = null;
    formVersion.value += 1;
  }

  function dispose() {
    clearSyncTimers();
    syncLock = null;
    lastEditSource = "toml";
    configId.value = null;
    isNew.value = false;
    selectedMeta.value = "";
    content.value = DEFAULT_CONFIG_TOML;
    form.value = null;
    schema.value = null;
    loading.value = false;
    saving.value = false;
    syncing.value = false;
    error.value = "";
    message.value = "";
    parseError.value = "";
    validationErrors.value = [];
    formVersion.value = 0;
    editorTab.value = "form";
    continuation.value = null;
    continuationSaveToLibrary.value = false;
    continuationLibraryId.value = "";
  }

  return {
    configId,
    selectedMeta,
    content,
    form,
    schema,
    modelCapabilities,
    loading,
    saving,
    syncing,
    error,
    message,
    parseError,
    validationErrors,
    formVersion,
    editorTab,
    continuation,
    continuationSaveToLibrary,
    continuationLibraryId,
    editingTitle,
    setContent,
    setForm,
    patchFormField,
    applyToml,
    flushSync,
    fetchSchema,
    newConfig,
    openConfig,
    loadContinuation,
    clearContinuation,
    saveExisting,
    createNew,
    validateConfig,
    queueContinuation,
    openFromRoute,
    bootstrapPickForJob,
    clearSelection,
    dispose,
  };
});

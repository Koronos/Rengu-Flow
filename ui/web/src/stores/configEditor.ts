import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { ElMessage } from "element-plus";
import type { RouteLocationNormalizedLoaded } from "vue-router";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { sanitizeConfigForm } from "../lib/configFormPayload";
import {
  getModelCapability,
  modelSupportsAdapters,
  pruneFormForModel,
} from "../lib/formUtils";
import { useTomlFormSync } from "../composables/useTomlFormSync";
import type { FormValues, ModelCapabilities } from "../types/forms";

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

export const useConfigEditorStore = defineStore("configEditor", () => {
  const configId = ref<string | null>(null);
  const isNew = ref(false);
  const selectedMeta = ref("");
  const content = ref(DEFAULT_CONFIG_TOML);
  const form = shallowRef<FormValues | null>({ _has_adapter: true });
  const schema = ref<Record<string, unknown> | null>(null);

  const loading = ref(false);
  const saving = ref(false);
  const syncing = ref(false);
  const error = ref("");
  const message = ref("");
  const parseError = ref("");
  const validationErrors = ref<string[]>([]);
  const formVersion = ref(0);
  const editorTab = ref("form");

  const continuation = ref<{ run_dir: string; resume_from: string } | null>(null);
  const continuationSaveToLibrary = ref(false);
  const continuationLibraryId = ref("");

  const modelCapabilities = computed(
    () => (schema.value?.registries as { model_capabilities?: ModelCapabilities })?.model_capabilities ?? {}
  );

  const editingTitle = computed(() => {
    if (isNew.value && !configId.value) return "New config";
    if (configId.value) return `Config #${configId.value}`;
    return "Config";
  });

  function cleanForm(raw: FormValues): FormValues | null {
    return sanitizeConfigForm(raw, modelCapabilities.value);
  }

  function applyModelCapabilityDefaults(target: FormValues): FormValues {
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
      if (!current || !allowed.includes(String(current))) {
        next["adapter.type"] = allowed[0];
        changed = true;
      }
    }

    return changed ? next : target;
  }

  const tomlSync = useTomlFormSync<FormValues>({
    content,
    form,
    syncing,
    parseError,
    requireNonemptyContent: true,
    sanitize: cleanForm,
    formatError,
    onFormVersionBump: () => {
      formVersion.value += 1;
    },
    transformParsed: applyModelCapabilityDefaults,
    parseToml: async (toml) => {
      const r = (await api.parseToml(toml)) as {
        ok?: boolean;
        form?: FormValues;
        error?: unknown;
      };
      return {
        ok: !!r.ok,
        form: r.form,
        error: r.error,
      };
    },
    renderToml: async (payload) => {
      const r = (await api.renderToml(payload)) as {
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
    schema.value = (await api.getSchema()) as Record<string, unknown>;
    return schema.value;
  }

  function setForm(nextForm: FormValues) {
    tomlSync.setForm(nextForm);
  }

  function patchFormField(path: string, value: unknown) {
    if (!form.value || !path) return;
    let next: FormValues = { ...form.value, [path]: value };
    if (path === "model.type") {
      next = pruneFormForModel(next, modelCapabilities.value);
      next = applyModelCapabilityDefaults(next);
    }
    setForm(next);
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
    await tomlSync.applyToml(DEFAULT_CONFIG_TOML);
  }

  async function openConfig(id: string) {
    resetEditorState();
    const data = (await api.getConfig(id)) as { content: string };
    configId.value = id;
    isNew.value = false;
    content.value = data.content;
    editorTab.value = "form";
    try {
      const summary = (await api.searchConfigs({ q: id, page: 1, page_size: 1 })) as {
        items?: { id: string; model_type?: string; dataset_ref?: string }[];
      };
      const row = summary.items?.find(
        (c) => c.id === id
      );
      if (row) {
        const parts: string[] = [];
        if (row.model_type) parts.push(row.model_type);
        if (row.dataset_ref) parts.push(row.dataset_ref);
        selectedMeta.value = parts.join(" · ");
      } else {
        selectedMeta.value = "";
      }
    } catch {
      selectedMeta.value = "";
    }
    await tomlSync.applyToml(data.content);
  }

  async function loadContinuation(runPath: string) {
    resetEditorState();
    const data = (await api.getRunConfig(runPath)) as {
      run_dir: string;
      resume_from: string;
      content: string;
    };
    continuation.value = {
      run_dir: data.run_dir,
      resume_from: data.resume_from,
    };
    continuationLibraryId.value = `${data.run_dir.split("/").pop()}_continued`;
    configId.value = null;
    isNew.value = true;
    selectedMeta.value = "from run folder";
    editorTab.value = "form";
    await tomlSync.applyToml(data.content);
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
      await tomlSync.flushSync();
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

  async function createNew(id: string) {
    const trimmed = (id || "").trim();
    if (!trimmed) return false;
    saving.value = true;
    error.value = "";
    message.value = "";
    try {
      await tomlSync.flushSync();
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

  async function validateConfig({ quiet = false }: { quiet?: boolean } = {}) {
    error.value = "";
    validationErrors.value = [];
    if (!quiet) message.value = "";
    try {
      await tomlSync.flushSync();
      const r = (await api.validate(content.value)) as {
        ok?: boolean;
        resolution?: Record<string, Record<string, unknown>>;
        errors?: string[];
        error?: string;
      };
      if (r.ok) {
        const res = r.resolution || {};
        const parts: string[] = [];
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
        return { ok: true as const, resolution: res };
      }
      validationErrors.value =
        Array.isArray(r.errors) && r.errors.length
          ? (r.errors as string[])
          : [String(r.error || "Invalid configuration.")];
      if (!quiet) {
        ElMessage.error(
          validationErrors.value.length === 1
            ? validationErrors.value[0]
            : `${validationErrors.value.length} issues — see list above`
        );
      }
      return { ok: false as const, errors: validationErrors.value };
    } catch (e) {
      error.value = formatError(e);
      if (!quiet) ElMessage.error(error.value);
      throw e;
    }
  }

  async function queueContinuation({
    startNow,
    saveToLibrary,
    libraryId,
  }: {
    startNow: boolean;
    saveToLibrary: boolean;
    libraryId?: string;
  }) {
    if (!continuation.value) return null;
    error.value = "";
    try {
      await tomlSync.flushSync();
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

  async function openFromRoute(route: RouteLocationNormalizedLoaded) {
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

  async function bootstrapPickForJob(storedConfigId: string) {
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
    tomlSync.resetSyncState();
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
    setContent: tomlSync.setContent,
    setForm,
    patchFormField,
    applyToml: tomlSync.applyToml,
    flushSync: tomlSync.flushSync,
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

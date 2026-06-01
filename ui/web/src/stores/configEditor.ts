import { defineStore } from "pinia";
import { computed, ref, shallowRef } from "vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import { sanitizeConfigForm } from "../lib/configFormPayload";
import {
  getModelCapability,
  modelSupportsAdapters,
  pruneFormForModel,
} from "../lib/formUtils";
import { applyOptimizerTypeChange } from "../lib/optimizerForm";
import { applySchedulerTypeChange } from "../lib/schedulerForm";
import { useTomlFormSync } from "../composables/useTomlFormSync";
import { createValidationAlertScheduler } from "../composables/useValidationAlertDismiss";
import type { FormValues, ModelCapabilities } from "../types/forms";

/** Offline fallback; keep in sync with rengu_flow_ui/templates/default_new_config.toml */
export const FALLBACK_DEFAULT_CONFIG_TOML = `dataset = ""

epochs = 1
gradient_accumulation_steps = 1
micro_batch_size_per_gpu = 1
logging_steps = 1
save_every_n_epochs = 1
output_dir = "output"

lr_scheduler = "cosine"
[lr_scheduler_args]
lr_min = 0.0

[model]
type = "sdxl"
dtype = "bfloat16"
checkpoint_path = ""

[adapter]
type = "lora"
rank = 16

[optimizer]
type = "adamw"
lr = 1.0e-4
`;

/**
 * Form/TOML editing state for a single run config, shared by RunFormModal. A run is
 * materialized only when added to the queue, so this store no longer tracks any standalone
 * "config library" identity — just the TOML content, the parsed form, validation, and the
 * optional `continuation` (resume-from-folder) context.
 */
export const useConfigEditorStore = defineStore("configEditor", () => {
  const defaultNewConfigToml = ref(FALLBACK_DEFAULT_CONFIG_TOML);
  const content = ref(FALLBACK_DEFAULT_CONFIG_TOML);
  const form = shallowRef<FormValues | null>({ _has_adapter: true });
  const schema = ref<Record<string, unknown> | null>(null);

  const loading = ref(false);
  const syncing = ref(false);
  const validating = ref(false);
  const error = ref("");
  const message = ref("");
  const parseError = ref("");
  const validationErrors = ref<string[]>([]);
  const formVersion = ref(0);

  /** Set when editing a run that resumes an existing run folder (continue training). */
  const continuation = ref<{ run_dir: string; resume_from: string } | null>(null);

  const validationAlertDismiss = createValidationAlertScheduler();

  const modelCapabilities = computed(
    () => (schema.value?.registries as { model_capabilities?: ModelCapabilities })?.model_capabilities ?? {}
  );

  const editorRunName = computed(() => {
    const raw = form.value?.run_name;
    return typeof raw === "string" && raw.trim() ? raw.trim() : "";
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
      const r = (await api.renderToml(payload, content.value)) as {
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

  function resolveDefaultNewConfigToml(fromSchema?: Record<string, unknown> | null): string {
    const raw = fromSchema?.default_new_config_toml;
    if (typeof raw === "string" && raw.trim()) {
      defaultNewConfigToml.value = raw;
      return raw;
    }
    return defaultNewConfigToml.value;
  }

  async function fetchSchema() {
    if (schema.value) return schema.value;
    schema.value = (await api.getSchema()) as Record<string, unknown>;
    resolveDefaultNewConfigToml(schema.value);
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
    if (path === "optimizer.type") {
      next = applyOptimizerTypeChange(next, value);
    }
    if (path === "lr_scheduler") {
      next = applySchedulerTypeChange(next, value);
    }
    setForm(next);
  }

  function resetEditorState() {
    validationErrors.value = [];
    error.value = "";
    message.value = "";
    parseError.value = "";
  }

  /** Reset to a blank new run (default template). */
  async function newConfig() {
    continuation.value = null;
    resetEditorState();
    await tomlSync.applyToml(resolveDefaultNewConfigToml(schema.value));
  }

  /** Load arbitrary TOML into the editor (seed from another run, or edit a draft/pending run). */
  async function loadContent(toml: string) {
    continuation.value = null;
    resetEditorState();
    await tomlSync.applyToml(toml || resolveDefaultNewConfigToml(schema.value));
  }

  /** Load a run's config from its on-disk folder and mark this as a continue-training run. */
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
    await tomlSync.applyToml(data.content);
  }

  function clearContinuation() {
    continuation.value = null;
  }

  async function validateConfig({ quiet = false }: { quiet?: boolean } = {}) {
    error.value = "";
    validationErrors.value = [];
    if (!quiet) {
      message.value = "";
      validationAlertDismiss.clearAll();
    }
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
          validationAlertDismiss.scheduleSuccessDismiss(() => {
            message.value = "";
          });
        }
        return { ok: true as const, resolution: res };
      }
      validationErrors.value =
        Array.isArray(r.errors) && r.errors.length
          ? (r.errors as string[])
          : [String(r.error || "Invalid configuration.")];
      if (!quiet) {
        validationAlertDismiss.scheduleErrorDismiss(() => {
          validationErrors.value = [];
        });
      }
      return { ok: false as const, errors: validationErrors.value };
    } catch (e) {
      error.value = formatError(e);
      if (!quiet) {
        validationAlertDismiss.scheduleErrorDismiss(() => {
          error.value = "";
        });
      }
      throw e;
    }
  }

  /**
   * Full pre-flight validation via the CLI config validator (`--validate-only`). This runs the
   * real pre-training checks (model rules, optimizer/scheduler, fused-optimizer vs grad-accum,
   * …) without loading datasets or model weights. On success it also calls the lightweight
   * `/validate` purely to build the optimizer/scheduler resolution summary.
   */
  async function validateFull(): Promise<{ ok: boolean }> {
    error.value = "";
    validationErrors.value = [];
    message.value = "";
    validationAlertDismiss.clearAll();
    validating.value = true;
    try {
      await tomlSync.flushSync();
      const result = await api.validateOnly(content.value);
      if (!result.ok) {
        validationErrors.value = [result.error || "Config validation failed."];
        validationAlertDismiss.scheduleErrorDismiss(() => {
          validationErrors.value = [];
        });
        return { ok: false };
      }
      // Reuse the lightweight validator just to build the nice resolution string.
      const parts: string[] = [];
      try {
        const r = (await api.validate(content.value)) as {
          ok?: boolean;
          resolution?: Record<string, Record<string, unknown>>;
        };
        const res = r.resolution || {};
        if (res.optimizer?.available) {
          parts.push(`optimizer → ${res.optimizer.resolved_class || res.optimizer.name}`);
        }
        if (res.scheduler?.available) {
          parts.push(
            `scheduler → ${res.scheduler.resolved || res.scheduler.resolved_class || res.scheduler.name}`
          );
        }
      } catch {
        // Resolution summary is best-effort; pre-flight already passed.
      }
      message.value = parts.length
        ? `Valid (${parts.join("; ")}) — pre-flight checks passed.`
        : "Valid — pre-flight checks passed.";
      validationAlertDismiss.scheduleSuccessDismiss(() => {
        message.value = "";
      });
      return { ok: true };
    } catch (e) {
      error.value = formatError(e);
      validationAlertDismiss.scheduleErrorDismiss(() => {
        error.value = "";
      });
      return { ok: false };
    } finally {
      validating.value = false;
    }
  }

  function clearValidationFeedback() {
    validationAlertDismiss.clearAll();
    validationErrors.value = [];
    message.value = "";
  }

  function clearValidationErrorBar() {
    validationAlertDismiss.clearAll();
    error.value = "";
  }

  function dispose() {
    tomlSync.resetSyncState();
    content.value = defaultNewConfigToml.value;
    form.value = null;
    loading.value = false;
    syncing.value = false;
    error.value = "";
    message.value = "";
    parseError.value = "";
    validationErrors.value = [];
    formVersion.value = 0;
    continuation.value = null;
    validationAlertDismiss.clearAll();
  }

  return {
    content,
    form,
    schema,
    modelCapabilities,
    loading,
    syncing,
    validating,
    error,
    message,
    parseError,
    validationErrors,
    formVersion,
    continuation,
    editorRunName,
    setContent: tomlSync.setContent,
    setForm,
    patchFormField,
    applyToml: tomlSync.applyToml,
    flushSync: tomlSync.flushSync,
    fetchSchema,
    newConfig,
    loadContent,
    loadContinuation,
    clearContinuation,
    validateConfig,
    validateFull,
    clearValidationFeedback,
    clearValidationErrorBar,
    dispose,
  };
});

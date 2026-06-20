/**
 * LR scheduler form helpers. Keep in sync with rengu_flow_ui/scheduler_form.py.
 *
 * Built-in scheduler names and KV defaults are NOT hardcoded here: they come from
 * the backend via `registries.scheduler_kv_defaults` (built-in names) and
 * `registries.scheduler_fqn_kv_defaults` (suggested torch FQNs). Passing them in
 * keeps custom-type detection and prefill in lock-step with the scheduler registry.
 */

import type { FormValues } from "../types/forms";

export const SCHEMA_SCHEDULER_PATHS = new Set([
  "lr_scheduler",
  "warmup_steps",
  "lr_scheduler_args.extra_params",
]);

/** Registry data the form needs, sourced from the schema's `registries`. */
export interface SchedulerRegistries {
  /** Lowercase built-in scheduler name -> default [lr_scheduler_args] KV rows. */
  builtinKvDefaults: Record<string, Record<string, unknown>>;
  /** Suggested fully-qualified scheduler path -> default KV rows. */
  fqnKvDefaults: Record<string, Record<string, unknown>>;
}

function builtinSet(registries: SchedulerRegistries | null | undefined): Set<string> {
  return new Set(Object.keys(registries?.builtinKvDefaults ?? {}).map((n) => n.toLowerCase()));
}

function schedulerKvDefaults(
  name: string,
  registries: SchedulerRegistries | null | undefined
): Record<string, unknown> {
  if (!name) return {};
  if (name.includes(".")) {
    return { ...(registries?.fqnKvDefaults?.[name] ?? {}) };
  }
  return { ...(registries?.builtinKvDefaults?.[name.toLowerCase()] ?? {}) };
}

export function normalizeSchedulerType(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value).trim();
}

export function isCustomSchedulerType(
  value: unknown,
  registries: SchedulerRegistries | null | undefined
): boolean {
  const name = normalizeSchedulerType(value);
  if (!name) return false;
  if (name.includes(".")) return true;
  return !builtinSet(registries).has(name.toLowerCase());
}

export function normalizeBuiltinSchedulerType(
  value: unknown,
  registries: SchedulerRegistries | null | undefined
): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (builtinSet(registries).has(trimmed.toLowerCase())) return trimmed.toLowerCase();
  return trimmed;
}

export function pathsRelevantForSchedulerType(schedType: unknown): Set<string> {
  const name = normalizeSchedulerType(schedType);
  if (!name || name.toLowerCase() === "none") {
    return new Set(["lr_scheduler"]);
  }
  return new Set(SCHEMA_SCHEDULER_PATHS);
}

export function pruneSchedulerForm(form: FormValues, schedType?: unknown): FormValues {
  const type = schedType !== undefined ? schedType : form["lr_scheduler"];
  const allowed = pathsRelevantForSchedulerType(type);
  const next = { ...form };
  for (const key of Object.keys(next)) {
    if (
      key === "lr_scheduler" ||
      key === "warmup_steps" ||
      key.startsWith("lr_scheduler_args.")
    ) {
      if (!allowed.has(key)) {
        delete next[key];
      }
    }
  }
  return next;
}

export function defaultsForSchedulerTypeChange(
  schedType: unknown,
  registries: SchedulerRegistries | null | undefined
): FormValues {
  const name = normalizeSchedulerType(schedType);
  if (!name) return {};
  return { "lr_scheduler_args.extra_params": schedulerKvDefaults(name, registries) };
}

export function applySchedulerTypeChange(
  form: FormValues,
  nextType: unknown,
  registries: SchedulerRegistries | null | undefined
): FormValues {
  const normalized = normalizeBuiltinSchedulerType(nextType, registries);
  let next: FormValues = { ...form, lr_scheduler: normalized };
  next = pruneSchedulerForm(next, normalized);
  const defaults = defaultsForSchedulerTypeChange(normalized, registries);
  if ("lr_scheduler_args.extra_params" in defaults) {
    next["lr_scheduler_args.extra_params"] = defaults["lr_scheduler_args.extra_params"];
  } else {
    delete next["lr_scheduler_args.extra_params"];
  }
  return next;
}

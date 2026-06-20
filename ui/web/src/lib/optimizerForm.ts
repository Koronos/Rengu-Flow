/**
 * Optimizer form helpers. Keep in sync with rengu_flow_ui/optimizer_form.py.
 *
 * The set of built-in optimizers and their KV defaults are NOT hardcoded here:
 * they come from the backend optimizer registry via `registries.optimizers` and
 * `registries.optimizer_kv_defaults` (exposed by config_schema.get_registries).
 * Passing them in keeps custom-type detection and prefill in lock-step with the
 * registry — a hand-maintained copy silently drifted and dropped defaults for
 * newer optimizers.
 */

import type { FormValues } from "../types/forms";

export const SCHEMA_OPTIMIZER_PATHS = new Set(["optimizer.type", "optimizer.extra_params"]);

/** Registry data the form needs, sourced from the schema's `registries`. */
export interface OptimizerRegistries {
  /** Lowercase built-in optimizer values (registry + aliases). */
  optimizers: string[];
  /** type -> default [optimizer] KV rows. */
  optimizerKvDefaults: Record<string, Record<string, unknown>>;
}

function builtinSet(registries: OptimizerRegistries | null | undefined): Set<string> {
  return new Set((registries?.optimizers ?? []).map((n) => n.toLowerCase()));
}

export function normalizeOptimizerType(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value).trim();
}

export function isCustomOptimizerType(
  value: unknown,
  registries: OptimizerRegistries | null | undefined
): boolean {
  const name = normalizeOptimizerType(value);
  if (!name) return false;
  if (name.includes(".")) return true;
  return !builtinSet(registries).has(name.toLowerCase());
}

export function normalizeBuiltinOptimizerType(
  value: unknown,
  registries: OptimizerRegistries | null | undefined
): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (builtinSet(registries).has(trimmed.toLowerCase())) return trimmed.toLowerCase();
  return trimmed;
}

export function pathsRelevantForOptimizerType(optType: unknown): Set<string> {
  const name = normalizeOptimizerType(optType);
  if (!name) return new Set(["optimizer.type"]);
  return new Set(SCHEMA_OPTIMIZER_PATHS);
}

export function pruneOptimizerForm(form: FormValues, optType?: unknown): FormValues {
  const type = optType !== undefined ? optType : form["optimizer.type"];
  const allowed = pathsRelevantForOptimizerType(type);
  const next = { ...form };
  for (const key of Object.keys(next)) {
    if (key.startsWith("optimizer.") && !allowed.has(key)) {
      delete next[key];
    }
  }
  return next;
}

export function defaultsForOptimizerTypeChange(
  optType: unknown,
  registries: OptimizerRegistries | null | undefined
): FormValues {
  const name = normalizeOptimizerType(optType);
  if (!name) return {};
  if (isCustomOptimizerType(name, registries)) {
    return { "optimizer.extra_params": {} };
  }
  const kv = registries?.optimizerKvDefaults?.[name.toLowerCase()] ?? {};
  return { "optimizer.extra_params": { ...kv } };
}

export function applyOptimizerTypeChange(
  form: FormValues,
  nextType: unknown,
  registries: OptimizerRegistries | null | undefined
): FormValues {
  const normalized = normalizeBuiltinOptimizerType(nextType, registries);
  let next: FormValues = { ...form, "optimizer.type": normalized };
  next = pruneOptimizerForm(next, normalized);
  const defaults = defaultsForOptimizerTypeChange(normalized, registries);
  if ("optimizer.extra_params" in defaults) {
    next["optimizer.extra_params"] = defaults["optimizer.extra_params"];
  } else {
    delete next["optimizer.extra_params"];
  }
  return next;
}

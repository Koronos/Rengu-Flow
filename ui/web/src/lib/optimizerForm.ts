/**
 * Optimizer form helpers. Keep in sync with rengu_flow_ui/optimizer_form.py.
 */

import type { FormValues } from "../types/forms";
import { optimizerExtraParamsDefaults } from "./optimKvDefaults";

export const SCHEMA_OPTIMIZER_PATHS = new Set(["optimizer.type", "optimizer.extra_params"]);

/** Lowercase registry + alias names (built-in selector values). */
export const KNOWN_BUILTIN_OPTIMIZER_TYPES = new Set([
  "adam",
  "adamw",
  "sgd",
  "adamw8bit",
  "adamw_optimi",
  "stableadamw",
  "offload",
  "genericoptim",
  "automagic",
  "adamw8bitkahan",
  "prodigy",
  // github.com/Koronos/K-Optimizers (installed on demand via the "koptim" profile).
  "adafusion",
  "muon",
]);

export function normalizeOptimizerType(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value).trim();
}

export function isCustomOptimizerType(value: unknown): boolean {
  const name = normalizeOptimizerType(value);
  if (!name) return false;
  if (name.includes(".")) return true;
  return !KNOWN_BUILTIN_OPTIMIZER_TYPES.has(name.toLowerCase());
}

export function normalizeBuiltinOptimizerType(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  const key = trimmed.toLowerCase();
  if (KNOWN_BUILTIN_OPTIMIZER_TYPES.has(key)) return key;
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

export function defaultsForOptimizerTypeChange(optType: unknown): FormValues {
  const name = normalizeOptimizerType(optType);
  if (!name) return {};
  if (isCustomOptimizerType(name)) {
    return { "optimizer.extra_params": {} };
  }
  return { "optimizer.extra_params": optimizerExtraParamsDefaults(name) };
}

export function applyOptimizerTypeChange(form: FormValues, nextType: unknown): FormValues {
  const normalized = normalizeBuiltinOptimizerType(nextType);
  let next: FormValues = { ...form, "optimizer.type": normalized };
  next = pruneOptimizerForm(next, normalized);
  const defaults = defaultsForOptimizerTypeChange(normalized);
  if ("optimizer.extra_params" in defaults) {
    next["optimizer.extra_params"] = defaults["optimizer.extra_params"];
  } else {
    delete next["optimizer.extra_params"];
  }
  return next;
}

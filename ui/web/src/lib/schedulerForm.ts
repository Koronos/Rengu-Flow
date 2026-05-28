/**
 * LR scheduler form helpers. Keep in sync with renga_flow_ui/scheduler_form.py.
 */

import type { FormValues } from "../types/forms";
import { schedulerKvDefaults } from "./optimKvDefaults";

export const SCHEMA_SCHEDULER_PATHS = new Set([
  "lr_scheduler",
  "warmup_steps",
  "lr_scheduler_args.extra_params",
]);

/** Lowercase built-in scheduler registry names. */
export const KNOWN_BUILTIN_SCHEDULER_TYPES = new Set([
  "constant",
  "linear",
  "cosine",
  "none",
]);

export function normalizeSchedulerType(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value).trim();
}

export function isCustomSchedulerType(value: unknown): boolean {
  const name = normalizeSchedulerType(value);
  if (!name) return false;
  if (name.includes(".")) return true;
  return !KNOWN_BUILTIN_SCHEDULER_TYPES.has(name.toLowerCase());
}

export function normalizeBuiltinSchedulerType(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  const key = trimmed.toLowerCase();
  if (KNOWN_BUILTIN_SCHEDULER_TYPES.has(key)) return key;
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

export function defaultsForSchedulerTypeChange(schedType: unknown): FormValues {
  const name = normalizeSchedulerType(schedType);
  if (!name) return {};
  return { "lr_scheduler_args.extra_params": schedulerKvDefaults(name) };
}

export function applySchedulerTypeChange(form: FormValues, nextType: unknown): FormValues {
  const normalized = normalizeBuiltinSchedulerType(nextType);
  let next: FormValues = { ...form, lr_scheduler: normalized };
  next = pruneSchedulerForm(next, normalized);
  const defaults = defaultsForSchedulerTypeChange(normalized);
  if ("lr_scheduler_args.extra_params" in defaults) {
    next["lr_scheduler_args.extra_params"] = defaults["lr_scheduler_args.extra_params"];
  } else {
    delete next["lr_scheduler_args.extra_params"];
  }
  return next;
}

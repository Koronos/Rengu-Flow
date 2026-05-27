/** Coerce form/TOML list field values into arrays (shared by *List parsers). */

import type { RawListInput } from "../types/forms";

export type { RawListInput };

export function coerceToArray(
  value: RawListInput,
  splitPattern: RegExp = /[,\s]+/
): unknown[] {
  if (typeof value === "number" && Number.isFinite(value)) {
    return [value];
  }
  let raw: unknown = value;
  if (typeof raw === "string" && raw.trim()) {
    const text = raw;
    try {
      raw = JSON.parse(text) as unknown;
    } catch {
      raw = text
        .split(splitPattern)
        .map((s) => s.trim())
        .filter(Boolean);
    }
  }
  return Array.isArray(raw) ? raw : [];
}

export function uniqueSorted<T>(items: T[], compare: (a: T, b: T) => number): T[] {
  return [...new Set(items)].sort(compare);
}

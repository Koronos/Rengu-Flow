/**
 * Key-value list helpers for optimizer/scheduler extra_params form fields.
 */

import { formatDefaultValue } from "./defaultFormat";

export type KvRow = { key: string; value: string };

export function formatKvValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return formatDefaultValue(value);
}

export function dictToKvRows(value: unknown): KvRow[] {
  if (value === null || value === undefined || value === "") return [];
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return [];
    try {
      return dictToKvRows(JSON.parse(trimmed) as unknown);
    } catch {
      return [];
    }
  }
  if (typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>).map(([key, v]) => ({
    key,
    value: formatKvValue(v),
  }));
}

/** Parse a single KV value string (numbers, booleans, JSON arrays/objects, or literal string). */
export function parseKvValue(raw: string): unknown {
  const text = raw.trim();
  if (!text) return "";
  if (text === "true") return true;
  if (text === "false") return false;
  if (text.startsWith("[") || text.startsWith("{")) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(text)) {
    const n = Number(text);
    if (!Number.isNaN(n)) return n;
  }
  return text;
}

export function kvRowsToDict(rows: KvRow[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const row of rows) {
    const key = row.key.trim();
    if (!key) continue;
    const parsed = parseKvValue(row.value);
    if (parsed === "" && !row.value.trim()) continue;
    out[key] = parsed;
  }
  return out;
}

export function isEmptyKvDict(value: unknown): boolean {
  const rows = dictToKvRows(value);
  return rows.length === 0;
}

/** Dict stored in the form after stripping incomplete KV rows. */
export function normalizeKvModelDict(value: unknown): Record<string, unknown> {
  return kvRowsToDict(dictToKvRows(value));
}

function kvDictsEqual(
  a: Record<string, unknown>,
  b: Record<string, unknown>
): boolean {
  const keysA = Object.keys(a).sort();
  const keysB = Object.keys(b).sort();
  if (keysA.length !== keysB.length) return false;
  return keysA.every(
    (key, i) => key === keysB[i] && JSON.stringify(a[key]) === JSON.stringify(b[key])
  );
}

/**
 * True when the parent form value is the same dict we would emit from local rows.
 * Used to avoid resetting the UI while the user is typing incomplete rows.
 */
export function kvModelMatchesRows(model: unknown, rows: KvRow[]): boolean {
  return kvDictsEqual(normalizeKvModelDict(model), kvRowsToDict(rows));
}

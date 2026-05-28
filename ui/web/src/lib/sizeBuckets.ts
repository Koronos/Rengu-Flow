/** Fixed [width, height, frames] training buckets (dataset TOML `size_buckets`). */

import { listToFormValue } from "./listToFormValue";

export type SizeBucket = [number, number, number];

export const SIZE_BUCKET_PRESETS: SizeBucket[] = [
  [512, 512, 1],
  [768, 768, 1],
  [1024, 1024, 1],
  [512, 768, 1],
  [768, 512, 1],
];

export function normalizeSizeBucket(row: unknown): SizeBucket | null {
  if (!Array.isArray(row) || row.length < 3) return null;
  const w = Number(row[0]);
  const h = Number(row[1]);
  const f = Number(row[2]);
  if (!Number.isFinite(w) || !Number.isFinite(h) || !Number.isFinite(f)) return null;
  if (w <= 0 || h <= 0 || f <= 0) return null;
  if (!Number.isInteger(w) || !Number.isInteger(h) || !Number.isInteger(f)) return null;
  return [w, h, f];
}

export function parseSizeBuckets(raw: unknown): SizeBucket[] {
  if (raw === undefined || raw === null || raw === "") return [];
  let parsed: unknown = raw;
  if (typeof raw === "string") {
    const text = raw.trim();
    if (!text) return [];
    try {
      parsed = JSON.parse(text);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(parsed)) return [];
  return parsed
    .map((row) => normalizeSizeBucket(row))
    .filter((row): row is SizeBucket => row !== null);
}

export function sizeBucketsFormValue(buckets: SizeBucket[]): SizeBucket[] | "" {
  return listToFormValue(buckets);
}

export function formatSizeBucketLabel(bucket: SizeBucket): string {
  return `${bucket[0]}×${bucket[1]}×${bucket[2]}`;
}

export function sizeBucketKey(bucket: SizeBucket): string {
  return `${bucket[0]}:${bucket[1]}:${bucket[2]}`;
}

export function validateSizeBucketsJson(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return "Invalid JSON";
  }
  if (!Array.isArray(parsed)) {
    return "Expected an array of [width, height, frames] rows";
  }
  if (parsed.length === 0) return null;
  for (let i = 0; i < parsed.length; i += 1) {
    const row = parsed[i];
    if (!Array.isArray(row) || row.length < 3) {
      return `Row ${i + 1}: expected [width, height, frames]`;
    }
    const normalized = normalizeSizeBucket(row);
    if (!normalized) {
      return `Row ${i + 1}: width, height, and frames must be positive integers`;
    }
  }
  return null;
}

export function sizeBucketsNeedJsonEditor(raw: unknown): boolean {
  if (raw === undefined || raw === null || raw === "") return false;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return false;
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed)) return true;
      return parsed.some((row) => normalizeSizeBucket(row) === null);
    } catch {
      return true;
    }
  }
  if (!Array.isArray(raw)) return true;
  return raw.some((row) => normalizeSizeBucket(row) === null);
}

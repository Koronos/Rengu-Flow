/** Training-config `renga-flow-dataset:` refs (optional `:label` suffix for TOML readability). */

import type { DatasetLibraryRefParts } from "../types/forms";

export const DATASET_REF_PREFIX = "renga-flow-dataset:";

export function parseDatasetLibraryRef(value: unknown): DatasetLibraryRefParts {
  const s = String(value ?? "").trim();
  if (!s.startsWith(DATASET_REF_PREFIX)) {
    return { isRef: false, id: null, label: null, canonical: s };
  }
  const rest = s.slice(DATASET_REF_PREFIX.length).trim();
  if (!rest) {
    return { isRef: true, id: null, label: null, canonical: DATASET_REF_PREFIX };
  }
  const colon = rest.indexOf(":");
  if (colon === -1) {
    const id = rest;
    return {
      isRef: true,
      id,
      label: null,
      canonical: `${DATASET_REF_PREFIX}${id}`,
    };
  }
  const id = rest.slice(0, colon).trim();
  const label = rest.slice(colon + 1).trim();
  return {
    isRef: true,
    id,
    label: label || null,
    canonical: `${DATASET_REF_PREFIX}${id}`,
  };
}

/** Alias aligned with Python `dataset_library_ref()`. */
export function formatDatasetLibraryRef(id: string | number, label?: string | null): string {
  const base = `${DATASET_REF_PREFIX}${id}`;
  const text = (label ?? "").trim();
  return text ? `${base}:${text}` : base;
}

export function canonicalDatasetRef(value: unknown): string {
  const p = parseDatasetLibraryRef(value);
  return p.isRef ? p.canonical : String(value ?? "").trim();
}

/** Tag / table label: display suffix, else library id, else full path. */
export function datasetRefDisplayLabel(value: unknown): string {
  const p = parseDatasetLibraryRef(value);
  if (p.isRef) {
    return p.label || p.id || String(value);
  }
  return String(value);
}

export function isLibraryDatasetRef(value: unknown): boolean {
  return parseDatasetLibraryRef(value).isRef;
}

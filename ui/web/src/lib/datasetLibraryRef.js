/** Training-config ``renga-flow-dataset:`` refs (optional ``:label`` suffix for TOML readability). */

export const DATASET_REF_PREFIX = "renga-flow-dataset:";

/**
 * @param {string} value
 * @returns {{ isRef: boolean, id: string | null, label: string | null, canonical: string }}
 */
export function parseDatasetLibraryRef(value) {
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

export function formatDatasetLibraryRef(id, label) {
  const base = `${DATASET_REF_PREFIX}${id}`;
  const text = (label ?? "").trim();
  return text ? `${base}:${text}` : base;
}

export function canonicalDatasetRef(value) {
  const p = parseDatasetLibraryRef(value);
  return p.isRef ? p.canonical : String(value ?? "").trim();
}

/** Tag / table label: display suffix, else library id, else full path. */
export function datasetRefDisplayLabel(value) {
  const p = parseDatasetLibraryRef(value);
  if (p.isRef) {
    return p.label || p.id || value;
  }
  return value;
}

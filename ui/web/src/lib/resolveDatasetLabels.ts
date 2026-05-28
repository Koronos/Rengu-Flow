import { api } from "../api";
import {
  canonicalDatasetRef,
  datasetRefDisplayLabel,
  libraryDatasetIdFromRef,
  parseDatasetLibraryRef,
} from "./datasetLibraryRef";

const labelCache = new Map<string, string>();

/** Sync label (cache / ref suffix / id); use {@link resolveDatasetDisplayLabel} for API lookup. */
export function peekDatasetDisplayLabel(value: unknown): string {
  const canon = canonicalDatasetRef(value);
  const cached = labelCache.get(canon);
  if (cached) return cached;
  return datasetRefDisplayLabel(value);
}

export function cacheDatasetDisplayLabel(value: unknown, label: string): void {
  const text = label.trim();
  if (!text) return;
  labelCache.set(canonicalDatasetRef(value), text);
}

/** Human-readable label; fetches library dataset name when ref has no `:label` suffix. */
export async function resolveDatasetDisplayLabel(value: unknown): Promise<string> {
  const parsed = parseDatasetLibraryRef(value);
  if (parsed.isRef && parsed.label) {
    const label = parsed.label;
    cacheDatasetDisplayLabel(value, label);
    return label;
  }

  const canon = canonicalDatasetRef(value);
  const cached = labelCache.get(canon);
  if (cached) return cached;

  const libraryId = libraryDatasetIdFromRef(value);
  if (libraryId) {
    try {
      const data = await api.getDataset(libraryId);
      const name =
        (data.meta?.name && String(data.meta.name).trim()) ||
        (typeof (data as { name?: string }).name === "string"
          ? String((data as { name?: string }).name).trim()
          : "");
      const label = name ? `${name} (#${libraryId})` : `Dataset #${libraryId}`;
      labelCache.set(canon, label);
      return label;
    } catch {
      const fallback = datasetRefDisplayLabel(value);
      return fallback;
    }
  }

  return datasetRefDisplayLabel(value);
}

export async function resolveDatasetDisplayLabels(
  values: string[]
): Promise<Map<string, string>> {
  const out = new Map<string, string>();
  const unique = [...new Set(values.filter((v) => v.trim()))];
  await Promise.all(
    unique.map(async (v) => {
      out.set(canonicalDatasetRef(v), await resolveDatasetDisplayLabel(v));
    })
  );
  return out;
}

import { ref } from "vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { DatasetScanPathResult } from "../types/api";

const CACHE_TTL_MS = 30_000;

type CacheEntry = { at: number; data: DatasetScanPathResult };

const cache = new Map<string, CacheEntry>();

function cacheKey(path: string): string {
  return path.trim();
}

export function useDatasetFolderStats() {
  const loading = ref(false);
  const error = ref("");
  const stats = ref<DatasetScanPathResult | null>(null);

  async function load(path: string, { force = false } = {}): Promise<void> {
    const key = cacheKey(path);
    if (!key) {
      stats.value = null;
      error.value = "No path set";
      return;
    }

    const hit = cache.get(key);
    if (!force && hit && Date.now() - hit.at < CACHE_TTL_MS) {
      stats.value = hit.data;
      error.value = hit.data.ok === false ? hit.data.error || "Path unavailable" : "";
      return;
    }

    loading.value = true;
    error.value = "";
    try {
      const data = await api.scanDatasetPath(key);
      stats.value = data;
      cache.set(key, { at: Date.now(), data });
      if (data.ok === false) {
        error.value = data.error || "Path unavailable";
      }
    } catch (e) {
      stats.value = null;
      error.value = formatError(e);
    } finally {
      loading.value = false;
    }
  }

  function clear(): void {
    stats.value = null;
    error.value = "";
    loading.value = false;
  }

  return { loading, error, stats, load, clear };
}

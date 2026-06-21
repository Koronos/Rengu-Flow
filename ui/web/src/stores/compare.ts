import { ref, watch } from "vue";
import { defineStore } from "pinia";
import type { CompareRunRow, TimelineEvent } from "../types/api";

// View settings persist to localStorage (survive navigation AND a full reload). The fetched run
// data is cached in-memory only (survives navigation): returning to /compare shows the last list
// instantly while a background refetch runs, instead of a blank "Loading runs…" flash.
const SETTINGS_KEY = "compare.settings";

interface Settings {
  smoothing: number;
  logScale: boolean;
  autoUpdate: boolean;
  cadenceSec: number;
}

function readSettings(): Partial<Settings> {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return JSON.parse(raw) as Partial<Settings>;
  } catch {
    /* private mode / corrupt value — fall back to defaults */
  }
  return {};
}

export const useCompareStore = defineStore("compare", () => {
  const s = readSettings();
  const smoothing = ref(typeof s.smoothing === "number" ? s.smoothing : 0);
  const logScale = ref(typeof s.logScale === "boolean" ? s.logScale : false);
  const autoUpdate = ref(typeof s.autoUpdate === "boolean" ? s.autoUpdate : false);
  const cadenceSec = ref(typeof s.cadenceSec === "number" ? s.cadenceSec : 10);

  watch([smoothing, logScale, autoUpdate, cadenceSec], () => {
    try {
      localStorage.setItem(
        SETTINGS_KEY,
        JSON.stringify({
          smoothing: smoothing.value,
          logScale: logScale.value,
          autoUpdate: autoUpdate.value,
          cadenceSec: cadenceSec.value,
        } satisfies Settings)
      );
    } catch {
      /* best-effort persistence */
    }
  });

  // Cached run data, keyed by the folder it was loaded for so a folder switch never shows stale rows.
  const allRuns = ref<CompareRunRow[]>([]);
  const metrics = ref<string[]>([]);
  const timelines = ref<Record<string, TimelineEvent[]>>({});
  const colorMap = ref<Record<string, string>>({});
  const selectedIds = ref<string[]>([]);
  const loadedDir = ref<string | null>(null);

  return {
    smoothing,
    logScale,
    autoUpdate,
    cadenceSec,
    allRuns,
    metrics,
    timelines,
    colorMap,
    selectedIds,
    loadedDir,
  };
});

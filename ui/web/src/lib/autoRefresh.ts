export type AutoRefreshIntervalSec = 0 | 5 | 10 | 30 | 60 | 120 | 300;

export const AUTO_REFRESH_STORAGE_KEY = "renga-flow.autoRefreshInterval";

export const AUTO_REFRESH_OPTIONS: { label: string; value: AutoRefreshIntervalSec }[] = [
  { label: "Off", value: 0 },
  { label: "5s", value: 5 },
  { label: "10s", value: 10 },
  { label: "30s", value: 30 },
  { label: "60s", value: 60 },
  { label: "2m", value: 120 },
  { label: "5m", value: 300 },
];

export const DEFAULT_AUTO_REFRESH_SEC: AutoRefreshIntervalSec = 10;

const ALLOWED = new Set<number>(AUTO_REFRESH_OPTIONS.map((o) => o.value));

export function parseStoredInterval(raw: string | null): AutoRefreshIntervalSec {
  if (raw == null || raw === "") return DEFAULT_AUTO_REFRESH_SEC;
  const n = Number(raw);
  if (ALLOWED.has(n)) return n as AutoRefreshIntervalSec;
  return DEFAULT_AUTO_REFRESH_SEC;
}

/** Poll interval in ms; 0 means off. Only polls when `isActive` (e.g. a live training run). */
export function effectiveRefreshMs(
  selectedSec: AutoRefreshIntervalSec,
  isActive = false
): number {
  if (selectedSec === 0 || !isActive) return 0;
  return selectedSec * 1000;
}

export const TRAIN_LIVE_REFRESH_STORAGE_KEY = "renga-flow.train-live-refresh";

export function formatLastUpdated(date: Date | null): string {
  if (!date) return "";
  return date.toLocaleTimeString();
}

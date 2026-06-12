/** Distinguishable, stable colors per run for the comparison overlay (sidebar swatch == chart series). */

// Tableau-10 + extras: high-contrast, color-blind-friendly-ish, distinct at small sizes.
export const RUN_COLOR_PALETTE = [
  "#4e79a7",
  "#f28e2b",
  "#59a14f",
  "#e15759",
  "#76b7b2",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
] as const;

export function colorForRun(index: number): string {
  const n = RUN_COLOR_PALETTE.length;
  return RUN_COLOR_PALETTE[((index % n) + n) % n];
}

/** Map run id -> color by position, so a run keeps its color regardless of which runs are selected. */
export function buildRunColorMap(runIds: string[]): Record<string, string> {
  const map: Record<string, string> = {};
  runIds.forEach((id, i) => {
    map[id] = colorForRun(i);
  });
  return map;
}

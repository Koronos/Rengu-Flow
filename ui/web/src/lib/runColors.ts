/** Distinguishable, stable colors per run for the comparison overlay (sidebar swatch == chart series). */

// Bright, saturated qualitative palette tuned to pop on the dark navy theme (the muted
// Tableau set read as dim on dark). Ordered for max distinctness at small sizes.
export const RUN_COLOR_PALETTE = [
  "#38bdf8", // sky
  "#fb923c", // orange
  "#4ade80", // green
  "#f87171", // red
  "#c084fc", // purple
  "#facc15", // yellow
  "#f472b6", // pink
  "#2dd4bf", // teal
  "#a3e635", // lime
  "#60a5fa", // blue
  "#fbbf24", // amber
  "#34d399", // emerald
  "#e879f9", // fuchsia
  "#fca5a5", // soft red
  "#93c5fd", // soft blue
  "#d8b4fe", // soft purple
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

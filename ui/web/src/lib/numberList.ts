import { coerceToArray, uniqueSorted, type RawListInput } from "./parseListInput";

/** Parse form values into a sorted unique list of numbers (floats). */

function roundNumber(n: number): number {
  const rounded = Math.round(n * 10000) / 10000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function parseNumberList(value: RawListInput, maxLength?: number): number[] {
  const raw = coerceToArray(value);
  if (maxLength !== undefined && maxLength >= 0) {
    const seen = new Set<number>();
    const ordered: number[] = [];
    for (const item of raw) {
      const n = Number.parseFloat(String(item).trim());
      if (!Number.isFinite(n)) continue;
      const rounded = roundNumber(n);
      if (seen.has(rounded)) continue;
      seen.add(rounded);
      ordered.push(rounded);
      if (ordered.length >= maxLength) break;
    }
    return ordered;
  }
  const out: number[] = [];
  for (const item of raw) {
    const n = Number.parseFloat(String(item).trim());
    if (!Number.isFinite(n)) continue;
    out.push(roundNumber(n));
  }
  return uniqueSorted(out, (a, b) => a - b);
}

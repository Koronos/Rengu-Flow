import { coerceToArray, uniqueSorted, type RawListInput } from "./parseListInput";

/** Parse form values into a sorted unique list of numbers (floats). */

function roundNumber(n: number): number {
  const rounded = Math.round(n * 10000) / 10000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function parseNumberList(value: RawListInput): number[] {
  const raw = coerceToArray(value);
  const out: number[] = [];
  for (const item of raw) {
    const n = Number.parseFloat(String(item).trim());
    if (!Number.isFinite(n)) continue;
    out.push(roundNumber(n));
  }
  return uniqueSorted(out, (a, b) => a - b);
}

import { coerceToArray, uniqueSorted, type RawListInput } from "./parseListInput";

/** Parse dataset/config form values into a sorted unique list of positive integers. */

export function parseIntegerList(value: RawListInput): number[] {
  const raw = coerceToArray(value);
  const out: number[] = [];
  for (const item of raw) {
    const n = Number.parseInt(String(item).trim(), 10);
    if (!Number.isFinite(n) || n <= 0) continue;
    out.push(n);
  }
  return uniqueSorted(out, (a, b) => a - b);
}

export function integerListToFormValue(numbers: number[]): number[] | "" {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    return "";
  }
  return numbers;
}

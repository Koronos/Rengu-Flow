import { coerceToArray, type RawListInput } from "./parseListInput";

/** Parse form values into a list of non-empty strings. */

export function parseStringList(value: RawListInput): string[] {
  const raw = coerceToArray(value, /\n/);
  const out: string[] = [];
  for (const item of raw) {
    if (typeof item === "string" && item.trim()) {
      out.push(item.trim());
    }
  }
  return out;
}

/** True when the value must stay as JSON (named prompt tables, etc.). */
export function stringListNeedsJsonEditor(value: RawListInput): boolean {
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.some((item) => typeof item !== "string");
      }
    } catch {
      return value.includes("{") || value.includes("name");
    }
  }
  const arr = coerceToArray(value, /\n/);
  return arr.some((item) => typeof item !== "string");
}

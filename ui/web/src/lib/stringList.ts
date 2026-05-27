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
  let raw: unknown = value;
  if (typeof raw === "string" && raw.trim()) {
    try {
      raw = JSON.parse(raw) as unknown;
    } catch {
      const s = raw as string;
      return s.includes("{") || s.includes("name");
    }
  }
  if (!Array.isArray(raw)) {
    return false;
  }
  return raw.some((item) => typeof item !== "string");
}

export function stringListToFormValue(strings: string[]): string[] | "" {
  if (!Array.isArray(strings) || strings.length === 0) {
    return "";
  }
  return strings;
}

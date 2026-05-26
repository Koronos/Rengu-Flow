/** Parse form values into a list of non-empty strings. */

export function parseStringList(value) {
  let raw = value;
  if (typeof raw === "string" && raw.trim()) {
    try {
      raw = JSON.parse(raw);
    } catch {
      raw = raw
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
    }
  }
  if (!Array.isArray(raw)) {
    return [];
  }
  const out = [];
  for (const item of raw) {
    if (typeof item === "string" && item.trim()) {
      out.push(item.trim());
    }
  }
  return out;
}

/** True when the value must stay as JSON (named prompt tables, etc.). */
export function stringListNeedsJsonEditor(value) {
  let raw = value;
  if (typeof raw === "string" && raw.trim()) {
    try {
      raw = JSON.parse(raw);
    } catch {
      return raw.includes("{") || raw.includes("name");
    }
  }
  if (!Array.isArray(raw)) {
    return false;
  }
  return raw.some((item) => typeof item !== "string");
}

export function stringListToFormValue(strings) {
  if (!Array.isArray(strings) || strings.length === 0) {
    return "";
  }
  return strings;
}

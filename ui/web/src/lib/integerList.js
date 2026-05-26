/** Parse dataset/config form values into a sorted unique list of positive integers. */

export function parseIntegerList(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    const n = Math.trunc(value);
    return n > 0 ? [n] : [];
  }
  let raw = value;
  if (typeof raw === "string" && raw.trim()) {
    try {
      raw = JSON.parse(raw);
    } catch {
      const part = raw
        .split(/[,\s]+/)
        .map((s) => s.trim())
        .filter(Boolean);
      raw = part;
    }
  }
  if (!Array.isArray(raw)) {
    return [];
  }
  const out = [];
  for (const item of raw) {
    const n = Number.parseInt(String(item).trim(), 10);
    if (!Number.isFinite(n) || n <= 0) continue;
    out.push(n);
  }
  return [...new Set(out)].sort((a, b) => a - b);
}

export function integerListToFormValue(numbers) {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    return "";
  }
  return numbers;
}

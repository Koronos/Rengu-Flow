/** Parse form values into a sorted unique list of numbers (floats). */

export function parseNumberList(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return [roundNumber(value)];
  }
  let raw = value;
  if (typeof raw === "string" && raw.trim()) {
    try {
      raw = JSON.parse(raw);
    } catch {
      raw = raw
        .split(/[,\s]+/)
        .map((s) => s.trim())
        .filter(Boolean);
    }
  }
  if (!Array.isArray(raw)) {
    return [];
  }
  const out = [];
  for (const item of raw) {
    const n = Number.parseFloat(String(item).trim());
    if (!Number.isFinite(n)) continue;
    out.push(n);
  }
  return [...new Set(out.map((n) => roundNumber(n)))].sort((a, b) => a - b);
}

function roundNumber(n) {
  const rounded = Math.round(n * 10000) / 10000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function numberListToFormValue(numbers) {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    return "";
  }
  return numbers;
}

/**
 * Format numeric defaults for UI hints and KV pre-fill (TOML-parseable strings).
 * Keep in sync with rengu_flow_ui/default_format.py.
 */

const SMALL_ABS_THRESHOLD = 1e-3;
const LARGE_ABS_THRESHOLD = 1e4;

export function formatScientific(n: number): string {
  if (n === 0) return "0";
  const sign = n < 0 ? "-" : "";
  const x = Math.abs(n);
  const exp = Math.floor(Math.log10(x));
  let mant = x / 10 ** exp;
  mant = Math.round(mant * 1e12) / 1e12;
  let mantStr: string;
  if (Math.abs(mant - Math.round(mant)) < 1e-9) {
    mantStr = String(Math.round(mant));
  } else {
    mantStr = mant.toFixed(12).replace(/\.?0+$/, "");
  }
  if (exp < 0) return `${sign}${mantStr}e${exp}`;
  if (exp > 0) return `${sign}${mantStr}e+${exp}`;
  return `${sign}${mantStr}`;
}

function fixedDecimalPlaces(n: number): { text: string; places: number } {
  if (n === 0) return { text: "0", places: 0 };
  const absN = Math.abs(n);
  const decimals = absN >= 1 ? 12 : Math.max(0, -Math.floor(Math.log10(absN))) + 6;
  let text = n.toFixed(decimals).replace(/\.?0+$/, "");
  if (!text.includes(".")) return { text, places: 0 };
  return { text, places: text.split(".")[1]?.length ?? 0 };
}

export function formatDefaultNumber(n: number): string {
  if (!Number.isFinite(n)) return String(n);
  if (n === 0) return "0";

  const absN = Math.abs(n);
  const useSci = absN >= LARGE_ABS_THRESHOLD || (absN > 0 && absN < SMALL_ABS_THRESHOLD);
  if (!useSci && Number.isInteger(n) && Math.abs(n) < 1e15) return String(Math.trunc(n));

  if (useSci) return formatScientific(n);

  const { text, places } = fixedDecimalPlaces(n);
  if (places > 3) return formatScientific(n);
  return text;
}

export function formatDefaultValue(val: unknown): string {
  if (typeof val === "boolean") return val ? "true" : "false";
  if (typeof val === "number") return formatDefaultNumber(val);
  if (Array.isArray(val) || (val !== null && typeof val === "object")) {
    return JSON.stringify(val);
  }
  return String(val ?? "");
}

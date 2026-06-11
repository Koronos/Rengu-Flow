// Per-resolution micro batch: the config value is either a plain integer
// (uniform batch for every bucket) or a { "512": 2, "1024": 1 } map keyed by
// resolution (TOML keys arrive as strings; the backend normalizes to int).
// This lib converts between that value and the widget's row model.

export interface MicroBatchRow {
  resolution: number | undefined;
  batch: number | undefined;
}

export type MicroBatchMode = "uniform" | "per_resolution";

export interface MicroBatchModel {
  mode: MicroBatchMode;
  uniform: number | undefined;
  rows: MicroBatchRow[];
}

function asPositiveInt(v: unknown): number | undefined {
  const n = Number(v);
  return Number.isInteger(n) && n > 0 ? n : undefined;
}

export function parseMicroBatch(value: unknown): MicroBatchModel {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const rows: MicroBatchRow[] = Object.entries(value as Record<string, unknown>).map(
      ([k, v]) => ({ resolution: asPositiveInt(k), batch: asPositiveInt(v) }),
    );
    rows.sort((a, b) => (a.resolution ?? Infinity) - (b.resolution ?? Infinity));
    return { mode: "per_resolution", uniform: undefined, rows };
  }
  return { mode: "uniform", uniform: asPositiveInt(value), rows: [] };
}

// Returns the config value for the model: an integer, a resolution map, or
// undefined when nothing usable is filled in (field left unset).
export function serializeMicroBatch(model: MicroBatchModel): number | Record<string, number> | undefined {
  if (model.mode === "uniform") {
    return model.uniform;
  }
  const out: Record<string, number> = {};
  for (const row of model.rows) {
    if (row.resolution !== undefined && row.batch !== undefined) {
      out[String(row.resolution)] = row.batch;
    }
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export function microBatchIssues(model: MicroBatchModel): string[] {
  if (model.mode === "uniform") return [];
  const issues: string[] = [];
  const seen = new Set<number>();
  for (const row of model.rows) {
    if (row.resolution === undefined || row.batch === undefined) {
      issues.push("Every row needs a resolution and a batch size.");
      break;
    }
  }
  for (const row of model.rows) {
    if (row.resolution !== undefined) {
      if (seen.has(row.resolution)) {
        issues.push(`Duplicate resolution ${row.resolution}.`);
      }
      seen.add(row.resolution);
    }
  }
  return issues;
}

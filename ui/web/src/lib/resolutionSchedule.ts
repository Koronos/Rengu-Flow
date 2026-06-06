/** Staged multi-resolution schedule (dataset TOML `[resolution_schedule]`).
 *
 * Shape on the wire (a TOML table, carried in the form as a JSON string):
 *   { enabled: bool, stage: [ { resolutions: number[], fraction: number }, ... ] }
 * The array-of-tables key is `stage` so `form_to_toml` emits
 * `[[resolution_schedule.stage]]` and the trainer reads it back.
 */

export interface ScheduleStage {
  resolutions: number[];
  fraction: number;
}

export interface ResolutionSchedule {
  enabled: boolean;
  stages: ScheduleStage[];
}

export function normalizeResolutions(value: unknown): number[] {
  const items = Array.isArray(value) ? value : value === undefined || value === null || value === ""
    ? []
    : [value];
  const out: number[] = [];
  for (const item of items) {
    const n = Number(item);
    if (Number.isFinite(n) && n > 0) out.push(Math.round(n));
  }
  return out;
}

export function normalizeStage(row: unknown): ScheduleStage | null {
  if (!row || typeof row !== "object") return null;
  const obj = row as Record<string, unknown>;
  const resolutions = normalizeResolutions(obj.resolutions ?? obj.resolution);
  const fraction = Number(obj.fraction);
  if (resolutions.length === 0) return null;
  if (!Number.isFinite(fraction) || fraction <= 0) return null;
  return { resolutions, fraction };
}

function coerceObject(raw: unknown): Record<string, unknown> | null {
  if (raw === undefined || raw === null || raw === "") return null;
  let parsed: unknown = raw;
  if (typeof raw === "string") {
    const text = raw.trim();
    if (!text) return null;
    try {
      parsed = JSON.parse(text);
    } catch {
      return null;
    }
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  return parsed as Record<string, unknown>;
}

export function parseResolutionSchedule(raw: unknown): ResolutionSchedule {
  const obj = coerceObject(raw);
  if (!obj) return { enabled: false, stages: [] };
  const rawStages = obj.stage ?? obj.stages;
  const stages: ScheduleStage[] = Array.isArray(rawStages)
    ? rawStages.map(normalizeStage).filter((s): s is ScheduleStage => s !== null)
    : [];
  return { enabled: Boolean(obj.enabled), stages };
}

export function scheduleFormValue(schedule: ResolutionSchedule): string | "" {
  const stages = schedule.stages
    .map(normalizeStage)
    .filter((s): s is ScheduleStage => s !== null);
  // Nothing meaningful to persist -> omit the key from TOML.
  if (!schedule.enabled && stages.length === 0) return "";
  return JSON.stringify({
    enabled: schedule.enabled,
    stage: stages.map((s) => ({ resolutions: s.resolutions, fraction: s.fraction })),
  });
}

/** Each stage's share of the run as a percent (fractions are normalized to sum 1). */
export function stageEffectivePercent(stages: ScheduleStage[]): number[] {
  const total = stages.reduce((sum, s) => sum + (s.fraction > 0 ? s.fraction : 0), 0);
  if (total <= 0) return stages.map(() => 0);
  return stages.map((s) => (s.fraction > 0 ? (s.fraction / total) * 100 : 0));
}

export function validateResolutionScheduleJson(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return "Invalid JSON";
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return "Expected an object like { enabled, stage: [...] }";
  }
  const obj = parsed as Record<string, unknown>;
  const rawStages = obj.stage ?? obj.stages;
  if (rawStages !== undefined && !Array.isArray(rawStages)) {
    return "`stage` must be an array of { resolutions, fraction }";
  }
  const list = Array.isArray(rawStages) ? rawStages : [];
  for (let i = 0; i < list.length; i += 1) {
    const stage = list[i];
    if (!stage || typeof stage !== "object") return `Stage ${i + 1}: expected an object`;
    const entry = stage as Record<string, unknown>;
    const res = normalizeResolutions(entry.resolutions ?? entry.resolution);
    if (res.length === 0) {
      return `Stage ${i + 1}: 'resolutions' must be a non-empty list of positive numbers`;
    }
    const fraction = Number(entry.fraction);
    if (!Number.isFinite(fraction) || fraction <= 0) {
      return `Stage ${i + 1}: 'fraction' must be a number > 0`;
    }
  }
  return null;
}

/** True when the value can't be represented in the table editor (malformed). */
export function scheduleNeedsJsonEditor(raw: unknown): boolean {
  if (raw === undefined || raw === null || raw === "") return false;
  const obj = coerceObject(raw);
  if (!obj) return true;
  const rawStages = obj.stage ?? obj.stages;
  if (rawStages !== undefined && !Array.isArray(rawStages)) return true;
  const list = Array.isArray(rawStages) ? rawStages : [];
  return list.some((s) => normalizeStage(s) === null);
}

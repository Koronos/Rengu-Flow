import type { TrainingSignalDef } from "../types/api";

export type SignalVariant = "primary" | "danger" | undefined;

export interface TrainingSignalGroup {
  label: string;
  items: Array<{
    id: string;
    label: string;
    hint?: string;
    variant?: SignalVariant;
  }>;
}

export const ACTIVE_JOB_STATES = ["running", "stopping"] as const;

export function groupSignalDefinitions(
  defs: TrainingSignalDef[],
  diskExportWait: boolean
): TrainingSignalGroup[] {
  const visible = defs.filter((def) => !def.disk_wait_only || diskExportWait);
  const groups = new Map<string, TrainingSignalGroup>();
  for (const def of visible) {
    const existing = groups.get(def.group);
    const item = {
      id: def.id,
      label: def.label,
      hint: def.hint,
      variant: def.variant,
    };
    if (existing) {
      existing.items.push(item);
    } else {
      groups.set(def.group, { label: def.group, items: [item] });
    }
  }
  return Array.from(groups.values());
}

export function jobSignalsAvailable(
  job?: { state?: string | null; signals_available?: boolean | null } | null
): boolean {
  if (job?.signals_available != null) return Boolean(job.signals_available);
  const state = job?.state;
  return state != null && (ACTIVE_JOB_STATES as readonly string[]).includes(state);
}

export function fsRunSignalsAvailable(
  run?: { signals_available?: boolean | null } | null
): boolean {
  return Boolean(run?.signals_available);
}

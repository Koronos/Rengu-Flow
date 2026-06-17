// Single source of truth for how run/job states map to Element Plus tag colors and
// human labels. Keeping this here (instead of per-view copies) keeps the state palette
// consistent across Runs, Studio, and Run detail.

export type TagType = "primary" | "success" | "warning" | "info" | "danger";

// Color semantics:
//   running/stopping → success (green)  · live and healthy
//   finished         → primary (cyan)   · completed, on-brand; the runs you compare
//   pending          → warning (amber)  · queued, waiting to start
//   new/stopped      → info (gray)      · idle/terminal, low emphasis
//   failed           → danger (red)     · error
export function runStateTag(state: string | undefined): TagType {
  if (state === "running" || state === "stopping") return "success";
  if (state === "finished") return "primary";
  if (state === "pending") return "warning";
  if (state === "failed") return "danger";
  // new, stopped, and anything unknown read as neutral.
  return "info";
}

export function runStateLabel(state: string | undefined): string {
  if (state === "new") return "Saved";
  if (state === "running") return "Running";
  if (state === "stopping") return "Stopping";
  if (state === "pending") return "Pending";
  if (state === "finished") return "Finished";
  if (state === "stopped") return "Stopped";
  if (state === "failed") return "Error";
  return String(state ?? "—");
}

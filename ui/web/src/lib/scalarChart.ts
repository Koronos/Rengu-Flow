/** Helpers for TensorBoard-style scalar line charts in the UI. */

export interface ScalarPoint {
  step: number;
  value: number;
  wall_time?: number;
}

export function nearestPointIndex(clientX: number, rect: DOMRect, pointCount: number): number {
  if (pointCount <= 0) return -1;
  if (pointCount === 1) return 0;
  const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
  const ratio = rect.width > 0 ? x / rect.width : 0;
  return Math.round(ratio * (pointCount - 1));
}

export function formatScalarValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.0001)) return value.toExponential(4);
  return value.toFixed(6);
}

export function formatWallTime(wallTime: number | undefined): string | null {
  if (wallTime == null || !Number.isFinite(wallTime)) return null;
  const ms = wallTime > 1e12 ? wallTime : wallTime * 1000;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString();
}

export const STEP_JUMP_SCALAR_TAGS = [
  "train/grad_norm",
  "train/prodigy_d",
  "train/automagic_avg_lr",
] as const;

export function resolveStepJumpTag(
  scalars: Record<string, ScalarPoint[]>
): (typeof STEP_JUMP_SCALAR_TAGS)[number] {
  for (const tag of STEP_JUMP_SCALAR_TAGS) {
    if (scalars[tag]?.length) return tag;
  }
  return "train/grad_norm";
}

export function stepJumpPanelTitle(tag: string): string {
  switch (tag) {
    case "train/prodigy_d":
      return "Prodigy D (adaptive step)";
    case "train/automagic_avg_lr":
      return "Avg learning rate";
    default:
      return "Gradient norm";
  }
}

/** Short tooltips for loss monitor panels (TensorBoard scalars). */
export const LOSS_PANEL_HINTS: Record<string, string> = {
  "train/epoch_loss":
    "Mean training loss aggregated per epoch from TensorBoard logs (train/epoch_loss).",
  "train/loss": "Per-step training loss (train/loss) — useful for spotting spikes or divergence.",
  preview:
    "Sample images from [preview] in config or from the Preview signal during training.",
  step_jump:
    "Extra scalar logged each step: gradient norm, Prodigy D, or Automagic average LR depending on optimizer.",
};

/** wandb-style debiased EMA smoothing for scalar series (applied client-side before charting). */

import type { ScalarMetricPoint } from "../types/api";

/**
 * Debiased exponential moving average (the same scheme wandb's smoothing slider uses).
 *
 * `weight` in [0, 1): 0 returns the series unchanged; higher = smoother. The debias term
 * (1 - weight^n) removes the cold-start bias so the smoothed curve starts at the first value and
 * a constant series stays constant. Non-finite values pass through untouched.
 */
export function emaSmooth(points: ScalarMetricPoint[], weight: number): ScalarMetricPoint[] {
  if (!(weight > 0) || points.length === 0) return points;
  const w = Math.min(0.999, weight);
  let last = 0;
  let numAccum = 0;
  return points.map((p) => {
    const v = p.value;
    if (!Number.isFinite(v)) return p;
    last = last * w + (1 - w) * v;
    numAccum += 1;
    const debias = 1 - Math.pow(w, numAccum);
    const smoothed = debias > 0 ? last / debias : v;
    return { ...p, value: smoothed };
  });
}

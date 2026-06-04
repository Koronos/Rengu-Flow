import type { RunProgress } from "../types/api";

/** Short speed + ETA hint for tables and live panel. */
export function formatRunProgressHint(progress: RunProgress | null | undefined): string {
  if (!progress) return "";
  const parts: string[] = [];
  // Prefer the EMA-smoothed s/it (kohya-style) so the speed reads steadily; fall back
  // to the instant step time, then to step/s.
  const sps = progress.steps_per_second_ema ?? progress.steps_per_second;
  const sit =
    progress.step_time_sec_ema ??
    progress.step_time_sec ??
    (sps != null && sps > 0 ? 1 / sps : null);
  if (sit != null && sit > 0) {
    parts.push(`${sit.toFixed(2)} s/it`);
  } else if (sps != null && sps > 0) {
    parts.push(`${sps.toFixed(2)} step/s`);
  }
  if (progress.eta) {
    parts.push(`ETA ${progress.eta}`);
  } else if (progress.steps_remaining != null && progress.max_steps) {
    parts.push(`${progress.steps_remaining} left`);
  }
  return parts.join(" · ");
}

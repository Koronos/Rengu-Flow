import type { RunProgress } from "../types/api";

/** Short speed + ETA hint for tables and live panel. */
export function formatRunProgressHint(progress: RunProgress | null | undefined): string {
  if (!progress) return "";
  const parts: string[] = [];
  // Prefer s/it (kohya-style) when we have a step time; otherwise show step/s.
  const sit = progress.step_time_sec;
  const sps = progress.steps_per_second_ema ?? progress.steps_per_second;
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

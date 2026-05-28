import type { RunProgress } from "../types/api";

/** Short speed + ETA hint for tables and live panel. */
export function formatRunProgressHint(progress: RunProgress | null | undefined): string {
  if (!progress) return "";
  const parts: string[] = [];
  const sps = progress.steps_per_second_ema ?? progress.steps_per_second;
  if (sps != null && sps > 0) {
    parts.push(`${sps.toFixed(2)} step/s`);
  }
  if (progress.eta) {
    parts.push(`ETA ${progress.eta}`);
  } else if (progress.steps_remaining != null && progress.max_steps) {
    parts.push(`${progress.steps_remaining} left`);
  }
  return parts.join(" · ");
}

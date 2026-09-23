/**
 * The registry preselect: which model(s) a stage starts with when the user has not chosen yet.
 *
 * One definition, read by the stage forms (`TagStageForm`, `CaptionStageForm`,
 * `EditCaptionStageForm`, when their registry loads) and by `workflowGraph.seedModelDefaults` (when
 * a workflow step is born, so a step added and run unopened carries the same model the form would
 * have shown). Each function answers only "what would the preselect pick"; **filling a gap, never
 * replacing a choice** stays with the caller, which is the one that knows whether a choice exists.
 */
import type { PrepModelInfo } from "../types/api";

/** Tag: every downloaded tagger, else the registry's first two (its default ensemble). */
export function preselectTagModels(registry: readonly PrepModelInfo[]): string[] {
  const downloaded = registry.filter((model) => model.downloaded).map((model) => model.id);
  return downloaded.length ? downloaded : registry.slice(0, 2).map((model) => model.id);
}

/** Caption / edit_caption: the registry's first model (its default), or `""` when it is empty. */
export function preselectModel(registry: readonly PrepModelInfo[]): string {
  return registry[0]?.id ?? "";
}

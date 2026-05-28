/** Helpers for training config ``preview.prompts`` (list + modal editor). */

export type PreviewEntryTable = {
  name?: string;
  prompt?: string;
  text?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  seed?: number;
  seed_stride?: number;
  preview_every_n_steps?: number;
  preview_every_n_epochs?: number;
  preview_offload_text_encoder?: boolean;
  preview_blocks_to_swap?: number;
  preview_save_png?: boolean;
  [key: string]: unknown;
};

export type PreviewEntry = string | PreviewEntryTable;

const ENTRY_OVERRIDE_KEYS: (keyof PreviewEntryTable)[] = [
  "negative_prompt",
  "width",
  "height",
  "num_inference_steps",
  "guidance_scale",
  "seed",
  "seed_stride",
  "preview_every_n_steps",
  "preview_every_n_epochs",
  "preview_offload_text_encoder",
  "preview_blocks_to_swap",
  "preview_save_png",
];

export function normalizePreviewEntries(raw: unknown): PreviewEntry[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((item) => item !== null && item !== undefined && item !== "");
}

export function previewEntryPrompt(entry: PreviewEntry): string {
  if (typeof entry === "string") return entry.trim();
  const p = (entry.prompt || entry.text || "").trim();
  return p;
}

export function previewEntryName(entry: PreviewEntry, index: number): string {
  if (typeof entry === "string") {
    const t = entry.trim();
    return t.length > 48 ? `${t.slice(0, 45)}…` : t || `Preview #${index + 1}`;
  }
  const name = (entry.name || "").trim();
  if (name) return name;
  const prompt = previewEntryPrompt(entry);
  if (prompt) {
    return prompt.length > 48 ? `${prompt.slice(0, 45)}…` : prompt;
  }
  return `Preview #${index + 1}`;
}

export function previewEntrySubtitle(entry: PreviewEntry, index: number): string {
  if (typeof entry === "string") {
    return `prompt_${index}`;
  }
  const tag = (entry.name || "").trim() || `prompt_${index}`;
  const prompt = previewEntryPrompt(entry);
  if (prompt && tag !== prompt) return prompt.length > 80 ? `${prompt.slice(0, 77)}…` : prompt;
  return tag;
}

export function countPreviewEntryOverrides(entry: PreviewEntry): number {
  if (typeof entry !== "object" || !entry) return 0;
  let n = 0;
  for (const key of ENTRY_OVERRIDE_KEYS) {
    const v = entry[key];
    if (v !== undefined && v !== null && v !== "") n += 1;
  }
  return n;
}

export function emptyPreviewEntryTable(): PreviewEntryTable {
  return { prompt: "" };
}

export function previewEntryToDraft(entry: PreviewEntry | null): PreviewEntryTable {
  if (!entry) return emptyPreviewEntryTable();
  if (typeof entry === "string") {
    return { prompt: entry };
  }
  return { ...entry };
}

export function clonePreviewEntry(entry: PreviewEntry): PreviewEntry {
  if (typeof entry === "string") return entry;
  return { ...entry };
}

export function duplicatePreviewEntry(entry: PreviewEntry): PreviewEntry {
  const draft = previewEntryToDraft(entry);
  const name = (draft.name || "").trim();
  if (name) {
    draft.name = `${name} (copy)`;
  }
  return serializePreviewEntry(draft);
}

function isEmptyValue(v: unknown): boolean {
  return v === undefined || v === null || v === "";
}

export function serializePreviewEntry(draft: PreviewEntryTable): PreviewEntry {
  const prompt = (draft.prompt || draft.text || "").trim();
  const name = (draft.name || "").trim();
  const overrides: PreviewEntryTable = {};
  for (const key of ENTRY_OVERRIDE_KEYS) {
    const v = draft[key];
    if (!isEmptyValue(v)) {
      (overrides as Record<string, unknown>)[key] = v;
    }
  }
  const hasOverrides = Object.keys(overrides).length > 0;
  if (!hasOverrides && !name) {
    return prompt;
  }
  const out: PreviewEntryTable = { ...overrides };
  if (name) out.name = name;
  if (prompt) out.prompt = prompt;
  else if (draft.text) out.text = String(draft.text).trim();
  return out;
}

export function previewEntryIsValid(entry: PreviewEntry): boolean {
  return Boolean(previewEntryPrompt(entry));
}

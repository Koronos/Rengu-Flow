/**
 * Pure helpers of the caption editor (`CaptionEditorView`): the textarea ↔ caption-lines mapping,
 * the dirty test, where "next"/"previous" lands across pages, the keyboard map, and the route
 * query every entry point builds (`/prep/captions?path=…&control_path=…`).
 */
import type { LocationQuery, RouteLocationRaw } from "vue-router";
import type { CaptionFormat } from "../types/api";

/** Textarea text → caption lines, exactly as the store writes them: trimmed, blank lines dropped. */
export function linesFromText(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export function textFromLines(lines: readonly string[]): string {
  return lines.join("\n");
}

/** Whether saving `draft` would change the caption (whitespace and blank lines do not count). */
export function isDraftDirty(saved: readonly string[], draft: string): boolean {
  const next = linesFromText(draft);
  return next.length !== saved.length || next.some((line, i) => line !== saved[i]);
}

export interface PageWindow {
  offset: number;
  limit: number;
  total: number;
  /** Items on the current page. */
  count: number;
}

/**
 * Where a step of `delta` (±1) from `index` on the current page lands: an index on this page, the
 * first item of the next page, the last item of the previous one, or `null` at either end.
 */
export type StepTarget =
  | { kind: "local"; index: number }
  | { kind: "page"; offset: number; at: "first" | "last" }
  | null;

export function stepTarget(index: number, delta: number, page: PageWindow): StepTarget {
  const next = index + delta;
  if (next >= 0 && next < page.count) return { kind: "local", index: next };
  if (next >= page.count && page.offset + page.count < page.total) {
    return { kind: "page", offset: page.offset + page.limit, at: "first" };
  }
  if (next < 0 && page.offset > 0) {
    return { kind: "page", offset: Math.max(0, page.offset - page.limit), at: "last" };
  }
  return null;
}

export type CaptionKeyAction = "save" | "next" | "prev" | null;

interface KeyLike {
  key: string;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  /** True when the event comes from a text field (plain arrows then move the caret). */
  inTextField?: boolean;
}

/**
 * Keyboard map: Ctrl/⌘+Enter saves; Alt+↓/↑ (anywhere) or ↓/↑ outside a text field move to the
 * next/previous image. Moving autosaves first.
 */
export function captionKeyAction(e: KeyLike): CaptionKeyAction {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) return "save";
  const arrow = e.key === "ArrowDown" ? "next" : e.key === "ArrowUp" ? "prev" : null;
  if (!arrow || e.ctrlKey || e.metaKey) return null;
  if (e.altKey || !e.inTextField) return arrow;
  return null;
}

export interface CaptionEditorTarget {
  path: string;
  control_path: string;
  format: CaptionFormat;
  ext: string;
}

const FORMATS: readonly CaptionFormat[] = ["sidecar", "json", "auto"];

function queryString(query: LocationQuery, key: string): string {
  const value = query[key];
  const first = Array.isArray(value) ? value[0] : value;
  return typeof first === "string" ? first : "";
}

export function parseCaptionEditorQuery(query: LocationQuery): CaptionEditorTarget {
  const format = queryString(query, "format") as CaptionFormat;
  return {
    path: queryString(query, "path"),
    control_path: queryString(query, "control_path"),
    format: FORMATS.includes(format) ? format : "sidecar",
    ext: queryString(query, "ext") || ".txt",
  };
}

/** Route to the caption editor on a folder; empty/default fields are left out of the URL. */
export function captionEditorLocation(target: Partial<CaptionEditorTarget> & { path: string }): RouteLocationRaw {
  const query: Record<string, string> = { path: target.path };
  if (target.control_path) query.control_path = target.control_path;
  if (target.format && target.format !== "sidecar") query.format = target.format;
  if (target.ext && target.ext !== ".txt" && target.format !== "json") query.ext = target.ext;
  return { name: "prep-captions", query };
}

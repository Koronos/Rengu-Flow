/** Heuristics for schema-driven and ad-hoc filesystem path fields. */

import type { SchemaField } from "../types/forms";
import { isLibraryDatasetRef } from "./datasetLibraryRef";

export type PathExpect = "file" | "dir";

const NON_PATH_FIELD_PATHS = new Set([
  "dataset",
  "eval_datasets",
  "activation_checkpointing",
  "reentrant_activation_checkpointing",
]);

const DIR_FIELD_PATHS = new Set([
  "path",
  "output_dir",
  "mask_path",
  "control_path",
]);

const FILE_FIELD_PATHS = new Set(["default_mask_file"]);

/** True when the field should show filesystem path validation. */
export function isPathField(field: SchemaField): boolean {
  const p = field.path || "";
  if (NON_PATH_FIELD_PATHS.has(p)) return false;
  if (field.type === "path") return true;
  if (DIR_FIELD_PATHS.has(p)) return true;
  if (FILE_FIELD_PATHS.has(p)) return true;
  if (p.endsWith("_dir")) return true;
  if (p.includes("_path") || p.endsWith("path")) return true;
  if (p.startsWith("resume_from")) return true;
  return false;
}

/** Expected node type for validation, or null to only check existence. */
export function pathFieldExpect(field: SchemaField): PathExpect | null {
  const p = field.path || "";
  if (DIR_FIELD_PATHS.has(p) || p.endsWith("_dir") || p === "output_dir") {
    return "dir";
  }
  if (FILE_FIELD_PATHS.has(p)) return "file";
  if (field.type === "path") return "file";
  if (p.includes("_path") || p.endsWith("path") || p.startsWith("resume_from")) {
    return "file";
  }
  return null;
}

/** Skip API checks for library refs and empty optional values. */
export function shouldSkipPathValidation(
  path: string,
  options?: { required?: boolean }
): boolean {
  const trimmed = (path || "").trim();
  if (!trimmed) return !options?.required;
  if (isLibraryDatasetRef(trimmed)) return true;
  return false;
}

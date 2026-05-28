import type { SchemaField } from "../types/forms";

/** Grid column span (24-col) for schema-driven form fields in config and dataset editors. */
export function schemaFieldColSpan(field: SchemaField): number {
  const path = field.path || "";
  if (
    field.type === "json" ||
    field.type === "key_value_list" ||
    field.type === "integer_list" ||
    field.type === "number_list"
  ) {
    return 24;
  }
  if (field.type === "boolean") {
    return 12;
  }
  if (
    path === "dataset" ||
    path === "eval_datasets" ||
    path === "output_dir" ||
    path.includes("path") ||
    path.endsWith("_dir")
  ) {
    return 24;
  }
  if (field.type === "string" && (field.allow_custom || (field.options?.length ?? 0) > 6)) {
    return 24;
  }
  return 12;
}

/** Dataset optional-section fields use a simpler layout than the main grid. */
export function schemaOptionalFieldColSpan(field: SchemaField): number {
  if (field.type === "json" || field.type === "key_value_list") return 24;
  return 12;
}

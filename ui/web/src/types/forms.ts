/** Shared form/schema types for training and dataset editors. */

export type FormValues = Record<string, unknown>;

export type RawListInput = string | number | unknown[] | null | undefined;

export interface VisibilityClause {
  all?: VisibilityClause[];
  any?: VisibilityClause[];
  not?: VisibilityClause;
  capability?: string;
  equals?: unknown;
  form_nonempty?: string;
  exclude_zero?: boolean;
  form_map_truthy?: { path: string; key: string };
  when_model_has_adapter?: boolean;
  field?: string;
  in?: unknown[];
}

export interface SchemaField {
  path: string;
  type: string;
  default?: unknown;
  visibility?: VisibilityClause;
  options?: unknown[];
  options_from_model?: boolean;
  option_values?: unknown[];
  allow_custom?: boolean;
  help?: string;
  doc_path?: string;
  label?: string;
  placeholder?: string;
  string_list_hint?: string;
  min?: number;
  max?: number;
  max_length?: number;
  runtime_tokens?: string[];
  importance?: string;
  recommended?: boolean;
  required?: boolean;
  show_if_set?: boolean;
  show_if_set_exclude_zero?: boolean;
  show_when_field?: string;
  when?: VisibilityClause;
  when_capability?: string;
  when_model_has_adapter?: boolean;
}

export interface ModelCapability {
  type_id: string;
  display_name?: string;
  branding_note?: string;
  aliases?: string[];
  features?: Record<string, boolean>;
  adapters?: string[];
  full_finetune?: boolean;
  model_fields?: { path: string; ui?: boolean }[];
}

export type ModelCapabilities = Record<string, ModelCapability>;

export interface DatasetLibraryRefParts {
  isRef: boolean;
  id: string | null;
  label: string | null;
  canonical: string;
}

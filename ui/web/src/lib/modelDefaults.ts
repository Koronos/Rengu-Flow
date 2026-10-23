/**
 * Per-model field defaults across a `model.type` switch.
 *
 * The schema emits one field per model (or group of models) for the paths whose default differs,
 * e.g. `preview.num_inference_steps` (sdxl 20, qwen_image21 28, moss_image 50), each gated by
 * `model.type`. A value in the form that still equals the *previous* model's default was never edited
 * (it was carried in by a template, a re-emitting control or the previous model's defaults): when the
 * model changes it must follow the newly visible field's default instead of pinning the old model's
 * number into the TOML. A value the user changed is kept.
 */
import { fieldVisible } from "./formUtils";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

/** Every field of every section of the schema. */
export function allSchemaFields(schema: Record<string, unknown> | null): SchemaField[] {
  const sections = (schema?.sections as { fields?: SchemaField[] }[] | undefined) ?? [];
  return sections.flatMap((s) => s.fields ?? []).filter((f) => typeof f.path === "string");
}

/** The default of the field with `path` that is visible for `form` (`undefined`: none visible or no default). */
function visibleDefault(
  fields: SchemaField[],
  path: string,
  form: FormValues,
  caps: ModelCapabilities | null
): { found: boolean; value: unknown } {
  const field = fields.find((f) => f.path === path && fieldVisible(f, form, caps || {}));
  if (!field) return { found: false, value: undefined };
  return { found: true, value: "default" in field ? field.default : undefined };
}

/**
 * `next` (the form after switching `model.type`) with every path that still holds the previous model's
 * default released to the new model's default (the key is dropped: the field shows, and the trainer
 * uses, the new model's value). Paths the user edited, and paths not visible for the new model, are kept.
 */
export function releaseStaleModelDefaults(
  prev: FormValues,
  next: FormValues,
  fields: SchemaField[],
  caps: ModelCapabilities | null = null
): FormValues {
  const out: FormValues = { ...next };
  const paths = new Set(fields.filter((f) => "default" in f).map((f) => f.path));
  for (const path of paths) {
    if (!(path in out) || path === "model.type") continue;
    const before = visibleDefault(fields, path, prev, caps);
    if (!before.found || before.value === undefined || out[path] !== before.value) continue;
    const after = visibleDefault(fields, path, out, caps);
    if (!after.found || after.value === before.value) continue;
    delete out[path];
  }
  return out;
}

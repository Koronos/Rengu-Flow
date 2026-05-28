import { isFormValueFilled } from "./formUtils";
import { isOverrideEnabled, setOverrideEnabled } from "./datasetDirectoryForm";
import type { FormValues, SchemaField } from "../types/forms";

interface DatasetFormSection {
  fields?: SchemaField[];
}

/** Root-level fields for the Dataset defaults tab (TOML root; not [[directory]]). */
export function sectionCoreFields(section: DatasetFormSection): SchemaField[] {
  return (section.fields || []).filter(
    (f) => !f.show_if_set && !f.show_when_field
  );
}

/** Root-level fields only when present in TOML or explicitly enabled. */
export function sectionOptionalFields(section: DatasetFormSection): SchemaField[] {
  return (section.fields || []).filter(
    (f) => f.show_if_set || f.show_when_field
  );
}

export { isOverrideEnabled, setOverrideEnabled, isFormValueFilled };

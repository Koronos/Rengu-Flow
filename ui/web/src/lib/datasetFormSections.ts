import { isFormValueFilled } from "./formUtils";
import { isOverrideEnabled, setOverrideEnabled } from "./datasetDirectoryForm";

/** Root-level fields for the Dataset defaults tab (TOML root; not [[directory]]). */
export function sectionCoreFields(section) {
  return (section.fields || []).filter(
    (f) => !f.show_if_set && !f.show_when_field
  );
}

/** Root-level fields only when present in TOML or explicitly enabled. */
export function sectionOptionalFields(section) {
  return (section.fields || []).filter(
    (f) => f.show_if_set || f.show_when_field
  );
}

export function optionalFieldActive(field, form) {
  return isOverrideEnabled(field, form);
}

export { isOverrideEnabled, setOverrideEnabled, isFormValueFilled };

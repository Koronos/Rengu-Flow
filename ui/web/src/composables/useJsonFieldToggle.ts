import { computed, ref, watch } from "vue";
import { jsonStringify } from "../lib/formUtils";

interface JsonFieldToggleOptions {
  /** Getter for the current field model value (string JSON or structured). */
  modelValue: () => unknown;
  /** Returns true when the value can only be edited as raw JSON (e.g. malformed). */
  needsJsonEditor: (value: unknown) => boolean;
  /** Validates the JSON text; returns an error message or null when valid. */
  validate: (text: string) => string | null;
  /** Emits the raw textarea value back to the parent (empty string when blank). */
  emit: (value: string) => void;
  /**
   * Require non-empty JSON before offering "back to table editor". Defaults to
   * true; pass false for fields whose empty state is still table-representable.
   */
  requireNonEmpty?: boolean;
}

/**
 * Shared "table editor ⇄ raw JSON" toggle used by the structured list fields
 * (size buckets, resolution schedule, tag-dropout rules). Encapsulates the
 * showJson state, the JSON text/error derivations, and the raw-input handler so
 * each field only has to supply its parse/validate helpers.
 */
export function useJsonFieldToggle(opts: JsonFieldToggleOptions) {
  const showJson = ref(false);

  watch(
    opts.modelValue,
    (value) => {
      if (opts.needsJsonEditor(value)) showJson.value = true;
    },
    { immediate: true }
  );

  const jsonText = computed(() => jsonStringify(opts.modelValue()));

  const jsonError = computed(() =>
    showJson.value ? opts.validate(jsonText.value) : null
  );

  const canUseTableEditor = computed(() => {
    if (jsonError.value) return false;
    return opts.requireNonEmpty === false || jsonText.value.trim() !== "";
  });

  function onJsonInput(text: string): void {
    opts.emit(text.trim() ? text : "");
  }

  return { showJson, jsonText, jsonError, canUseTableEditor, onJsonInput };
}

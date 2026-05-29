/**
 * Element Plus component prop types don't declare ARIA attributes, so binding
 * `aria-label="…"` directly trips vue-tsc. Spreading a record via `v-bind`
 * forwards them as fallthrough attrs (the runtime behaviour) without the
 * per-prop type check.
 */
export function ariaLabel(label: string): Record<string, string> {
  return { "aria-label": label };
}

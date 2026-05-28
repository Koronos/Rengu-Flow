/** Empty list fields serialize as "" in the form/API round-trip. */
export function listToFormValue<T>(items: T[]): T[] | "" {
  if (!Array.isArray(items) || items.length === 0) {
    return "";
  }
  return items;
}

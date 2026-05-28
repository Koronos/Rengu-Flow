/** Short display for ISO timestamps from the config/dataset library API. */
export function formatLibraryTimestamp(iso: string | null | undefined): string {
  if (!iso) return "";
  return String(iso).slice(0, 16).replace("T", " ");
}

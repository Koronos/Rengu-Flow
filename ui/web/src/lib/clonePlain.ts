/** Deep-clone plain JSON-safe values for form sanitizers. */
export function clonePlain<T>(raw: T): T {
  try {
    return structuredClone(raw);
  } catch {
    return JSON.parse(JSON.stringify(raw)) as T;
  }
}

/** Trigger a browser download from a Blob. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Parse filename from Content-Disposition (attachment; filename="…"). */
export function filenameFromContentDisposition(header: string | null): string | null {
  if (!header) return null;
  const quoted = /filename\*?=(?:UTF-8''|")?([^";\n]+)/i.exec(header);
  if (!quoted?.[1]) return null;
  try {
    return decodeURIComponent(quoted[1].replace(/"/g, "").trim());
  } catch {
    return quoted[1].replace(/"/g, "").trim();
  }
}

import { api } from "../api";

/**
 * @typedef {{ kind: 'library', id: string | number }} LibraryThumbSource
 * @typedef {{ kind: 'path', path: string }} PathThumbSource
 * @typedef {LibraryThumbSource | PathThumbSource} ThumbSource
 */

/** @param {ThumbSource | null | undefined} source */
export async function loadPreviewThumbs(source, limit = 4) {
  if (!source) return [];
  if (source.kind === "library") {
    return loadLibraryThumbs(source.id, limit);
  }
  if (source.kind === "path") {
    return loadPathThumbs(source.path, limit);
  }
  return [];
}

async function loadLibraryThumbs(libraryId, limit) {
  if (!libraryId) return [];
  try {
    const { content } = await api.getDataset(libraryId);
    const r = await api.listDatasetPreviewImages({ content, limit, offset: 0 });
    if (!r.ok) return [];
    return (r.images || []).slice(0, limit).map((img) => api.datasetPreviewImageUrl(img.token));
  } catch {
    return [];
  }
}

async function loadPathThumbs(path, limit) {
  const trimmed = (path || "").trim();
  if (!trimmed) return [];
  try {
    const r = await api.scanDatasetPath(trimmed);
    if (!r.ok) return [];
    const tokens = r.preview_tokens?.length
      ? r.preview_tokens
      : r.preview_token
        ? [r.preview_token]
        : [];
    return tokens.slice(0, limit).map((t) => api.datasetPreviewImageUrl(t));
  } catch {
    return [];
  }
}

/** @param {string | number} libraryId */
export function libraryThumbSource(libraryId) {
  return { kind: "library", id: libraryId };
}

/** @param {string} path */
export function pathThumbSource(path) {
  return { kind: "path", path: path || "" };
}

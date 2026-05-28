import { api } from "../api";
import { libraryDatasetIdFromRef } from "./datasetLibraryRef";

export type LibraryThumbSource = { kind: "library"; id: string | number };
export type PathThumbSource = { kind: "path"; path: string };
export type ThumbSource = LibraryThumbSource | PathThumbSource;

export async function loadPreviewThumbs(
  source: ThumbSource | null | undefined,
  limit = 4
): Promise<string[]> {
  if (!source) return [];
  if (source.kind === "library") {
    return loadLibraryThumbs(source.id, limit);
  }
  if (source.kind === "path") {
    return loadPathThumbs(source.path, limit);
  }
  return [];
}

async function loadLibraryThumbs(libraryId: string | number, limit: number): Promise<string[]> {
  if (!libraryId) return [];
  try {
    const { content } = (await api.getDataset(String(libraryId))) as { content: string };
    const r = (await api.listDatasetPreviewImages({ content, limit, offset: 0 })) as {
      ok?: boolean;
      images?: { token: string }[];
    };
    if (!r.ok) return [];
    return (r.images || []).slice(0, limit).map((img) => api.datasetPreviewImageUrl(img.token));
  } catch {
    return [];
  }
}

async function loadPathThumbs(path: string, limit: number): Promise<string[]> {
  const trimmed = (path || "").trim();
  if (!trimmed) return [];
  try {
    const r = (await api.scanDatasetPath(trimmed)) as {
      ok?: boolean;
      preview_tokens?: string[];
      preview_token?: string;
    };
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

export function libraryThumbSource(libraryId: string | number): LibraryThumbSource {
  return { kind: "library", id: libraryId };
}

export function pathThumbSource(path: string): PathThumbSource {
  return { kind: "path", path };
}

/** Map a config's indexed `dataset_ref` to a lazy-load thumb source (library id or folder/TOML path). */
export function datasetRefToThumbSource(
  datasetRef: string | null | undefined
): ThumbSource | null {
  const trimmed = (datasetRef ?? "").trim();
  if (!trimmed) return null;
  const libraryId = libraryDatasetIdFromRef(trimmed);
  if (libraryId) return libraryThumbSource(libraryId);
  return pathThumbSource(trimmed);
}

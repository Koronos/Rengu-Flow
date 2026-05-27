import { loadPreviewThumbs, pathThumbSource } from "./previewThumbs";

export { pathThumbSource };

/** @deprecated Prefer loadPreviewThumbs(pathThumbSource(path), limit). */
export function loadDirectoryPathThumbs(path, limit = 4) {
  return loadPreviewThumbs(pathThumbSource(path), limit);
}

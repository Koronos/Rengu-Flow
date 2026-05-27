import { libraryThumbSource, loadPreviewThumbs } from "./previewThumbs";

export { libraryThumbSource };

/** @deprecated Prefer loadPreviewThumbs(libraryThumbSource(id), limit). */
export function loadDatasetLibraryThumbs(libraryId, limit = 4) {
  return loadPreviewThumbs(libraryThumbSource(libraryId), limit);
}

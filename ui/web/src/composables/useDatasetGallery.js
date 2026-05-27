import { ref } from "vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";

/** Shared state for DatasetGalleryDialog across list / picker / editor sections. */
export function useDatasetGallery() {
  const galleryOpen = ref(false);
  const galleryTitle = ref("Image gallery");
  const galleryContent = ref("");
  const galleryDirectoryIndex = ref(null);
  const galleryLoading = ref(false);

  function showFromContent({ title, content, directoryIndex = null }) {
    galleryTitle.value = title || "Image gallery";
    galleryContent.value = content || "";
    galleryDirectoryIndex.value = directoryIndex;
    galleryOpen.value = true;
  }

  async function showFromLibrary({ id, title, directoryIndex = null }) {
    galleryTitle.value = title || `Dataset #${id}`;
    galleryDirectoryIndex.value = directoryIndex;
    galleryOpen.value = true;
    galleryLoading.value = true;
    galleryContent.value = "";
    try {
      const row = await api.getDataset(id);
      galleryContent.value = row.content || "";
    } catch (e) {
      galleryContent.value = "";
      galleryTitle.value = formatError(e);
    } finally {
      galleryLoading.value = false;
    }
  }

  return {
    galleryOpen,
    galleryTitle,
    galleryContent,
    galleryDirectoryIndex,
    galleryLoading,
    showFromContent,
    showFromLibrary,
  };
}

import { ref } from "vue";
import { defineStore } from "pinia";
import { api } from "../api";
import { formatError } from "../lib/formatError";

/** Shared state for the app-wide DatasetGalleryHost (list / picker / editor). */
export const useDatasetGalleryStore = defineStore("datasetGallery", () => {
  const galleryOpen = ref(false);
  const galleryTitle = ref("Image gallery");
  const galleryContent = ref("");
  const galleryDirectoryIndex = ref<number | undefined>(undefined);
  const galleryLoading = ref(false);
  const galleryError = ref("");

  function showFromContent({
    title,
    content,
    directoryIndex,
  }: {
    title?: string;
    content?: string;
    directoryIndex?: number | null;
  }) {
    galleryTitle.value = title || "Image gallery";
    galleryContent.value = content || "";
    galleryError.value = "";
    galleryDirectoryIndex.value = directoryIndex ?? undefined;
    galleryOpen.value = true;
  }

  async function showFromLibrary({
    id,
    title,
    directoryIndex,
  }: {
    id: string | number;
    title?: string;
    directoryIndex?: number | null;
  }) {
    galleryTitle.value = title || `Dataset #${id}`;
    galleryDirectoryIndex.value = directoryIndex ?? undefined;
    galleryOpen.value = true;
    galleryLoading.value = true;
    galleryContent.value = "";
    galleryError.value = "";
    try {
      const row = (await api.getDataset(String(id))) as { content?: string };
      galleryContent.value = row.content || "";
    } catch (e) {
      galleryContent.value = "";
      galleryError.value = formatError(e);
    } finally {
      galleryLoading.value = false;
    }
  }

  function closeGallery() {
    galleryOpen.value = false;
  }

  return {
    galleryOpen,
    galleryTitle,
    galleryContent,
    galleryDirectoryIndex,
    galleryLoading,
    galleryError,
    showFromContent,
    showFromLibrary,
    closeGallery,
  };
});

import { ref } from "vue";
import { defineStore } from "pinia";

/** App-wide single lightbox for dataset gallery thumbs (avoids stacked el-image-viewers). */
export const useDatasetImageViewerStore = defineStore("datasetImageViewer", () => {
  const viewerOpen = ref(false);
  const viewerUrls = ref<string[]>([]);
  const viewerIndex = ref(0);

  function openDatasetImageViewer(urls: string[], index: number) {
    if (!urls.length || index < 0 || index >= urls.length) return;
    viewerUrls.value = urls;
    viewerIndex.value = index;
    viewerOpen.value = true;
  }

  function closeDatasetImageViewer() {
    viewerOpen.value = false;
  }

  return {
    viewerOpen,
    viewerUrls,
    viewerIndex,
    openDatasetImageViewer,
    closeDatasetImageViewer,
  };
});

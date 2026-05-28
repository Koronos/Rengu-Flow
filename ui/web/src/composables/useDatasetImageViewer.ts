import { ref } from "vue";

/** App-wide single lightbox for dataset gallery thumbs (avoids stacked el-image-viewers). */
const viewerOpen = ref(false);
const viewerUrls = ref<string[]>([]);
const viewerIndex = ref(0);

export function useDatasetImageViewer() {
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
}

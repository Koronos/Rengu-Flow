import { onUnmounted, ref, unref, watch } from "vue";
import { loadPreviewThumbs } from "../lib/previewThumbs";

/**
 * Load preview image URLs from a library id or folder path.
 * @param {import('vue').MaybeRefOrGetter<import('../lib/previewThumbs').ThumbSource | null | undefined>} source
 * @param {import('vue').MaybeRefOrGetter<number>} [limit]
 */
export function usePreviewThumbs(source, limit = 4) {
  const thumbs = ref([]);
  const loading = ref(false);
  let requestId = 0;

  async function load() {
    const src = typeof source === "function" ? source() : unref(source);
    const lim = typeof limit === "function" ? limit() : unref(limit);
    const id = ++requestId;
    loading.value = true;
    try {
      const urls = await loadPreviewThumbs(src, lim);
      if (id === requestId) thumbs.value = urls;
    } catch {
      if (id === requestId) thumbs.value = [];
    } finally {
      if (id === requestId) loading.value = false;
    }
  }

  watch(() => [unref(source), unref(limit)], load, { immediate: true, deep: true });

  onUnmounted(() => {
    requestId += 1;
  });

  return { thumbs, loading };
}

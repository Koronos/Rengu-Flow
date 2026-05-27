import { onUnmounted, ref, unref, watch, type MaybeRefOrGetter, toValue } from "vue";
import { loadPreviewThumbs, type ThumbSource } from "../lib/previewThumbs";

/** Load preview image URLs from a library id or folder path. */
export function usePreviewThumbs(
  source: MaybeRefOrGetter<ThumbSource | null | undefined>,
  limit: MaybeRefOrGetter<number> = 4
) {
  const thumbs = ref<string[]>([]);
  const loading = ref(false);
  let requestId = 0;

  async function load() {
    const src = toValue(source);
    const lim = toValue(limit);
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

  watch(() => [toValue(source), toValue(limit)], load, { immediate: true, deep: true });

  onUnmounted(() => {
    requestId += 1;
  });

  return { thumbs, loading };
}

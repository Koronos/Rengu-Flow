import { onUnmounted, ref, type MaybeRefOrGetter, toValue, watch } from "vue";
import { loadPreviewThumbs, type ThumbSource } from "../lib/previewThumbs";

const DEBOUNCE_MS = 200;

function thumbSourceKey(src: ThumbSource | null | undefined): string {
  if (!src) return "";
  return src.kind === "library" ? `library:${src.id}` : `path:${src.path}`;
}

/** Load preview image URLs from a library id or folder path. */
export function usePreviewThumbs(
  source: MaybeRefOrGetter<ThumbSource | null | undefined>,
  limit: MaybeRefOrGetter<number> = 4
) {
  const thumbs = ref<string[]>([]);
  const loading = ref(false);
  let requestId = 0;
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;
  let lastWatchKey = "";

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

  function scheduleLoad() {
    const watchKey = `${thumbSourceKey(toValue(source))}\0${toValue(limit)}`;
    if (watchKey === lastWatchKey) {
      return;
    }
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      lastWatchKey = watchKey;
      load();
    }, DEBOUNCE_MS);
  }

  watch(() => `${thumbSourceKey(toValue(source))}\0${toValue(limit)}`, scheduleLoad, {
    immediate: true,
  });

  onUnmounted(() => {
    requestId += 1;
    if (debounceTimer) clearTimeout(debounceTimer);
  });

  return { thumbs, loading };
}

import { ref, type MaybeRefOrGetter, toValue, watch } from "vue";
import { loadPreviewThumbs, type ThumbSource } from "../lib/previewThumbs";
import { useLatestAsync } from "./useLatestAsync";

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
  const guard = useLatestAsync();
  let lastWatchKey = "";

  async function load() {
    const src = toValue(source);
    const lim = toValue(limit);
    const token = guard.begin();
    loading.value = true;
    try {
      const urls = await loadPreviewThumbs(src, lim);
      if (guard.isCurrent(token)) thumbs.value = urls;
    } catch {
      if (guard.isCurrent(token)) thumbs.value = [];
    } finally {
      if (guard.isCurrent(token)) loading.value = false;
    }
  }

  function scheduleLoad() {
    const watchKey = `${thumbSourceKey(toValue(source))}\0${toValue(limit)}`;
    if (watchKey === lastWatchKey) {
      return;
    }
    guard.schedule(() => {
      lastWatchKey = watchKey;
      load();
    }, DEBOUNCE_MS);
  }

  watch(() => `${thumbSourceKey(toValue(source))}\0${toValue(limit)}`, scheduleLoad, {
    immediate: true,
  });

  return { thumbs, loading };
}

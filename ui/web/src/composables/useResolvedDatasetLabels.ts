import { ref, watch, type Ref } from "vue";
import { canonicalDatasetRef, datasetRefDisplayLabel } from "../lib/datasetLibraryRef";
import {
  peekDatasetDisplayLabel,
  resolveDatasetDisplayLabels,
} from "../lib/resolveDatasetLabels";

/** Reactive display labels for dataset refs/paths (tags, chips). */
export function useResolvedDatasetLabels(paths: Ref<string[]>) {
  const labelsByCanon = ref<Record<string, string>>({});

  watch(
    paths,
    async (list) => {
      const keys = list.filter((p) => p.trim());
      if (!keys.length) {
        labelsByCanon.value = {};
        return;
      }
      const resolved = await resolveDatasetDisplayLabels(keys);
      const next: Record<string, string> = {};
      for (const p of keys) {
        const canon = canonicalDatasetRef(p);
        next[canon] = resolved.get(canon) ?? peekDatasetDisplayLabel(p);
      }
      labelsByCanon.value = next;
    },
    { immediate: true }
  );

  function labelFor(path: string): string {
    const canon = canonicalDatasetRef(path);
    return labelsByCanon.value[canon] ?? datasetRefDisplayLabel(path);
  }

  return { labelFor };
}

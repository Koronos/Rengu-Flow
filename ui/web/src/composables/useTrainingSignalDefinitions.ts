import { onMounted, ref } from "vue";
import { api } from "../api";
import type { TrainingSignalDef } from "../types/api";

let cachedDefs: TrainingSignalDef[] | null = null;
let loadPromise: Promise<TrainingSignalDef[]> | null = null;

async function loadSignalDefinitions(): Promise<TrainingSignalDef[]> {
  if (cachedDefs) return cachedDefs;
  if (!loadPromise) {
    loadPromise = api
      .listSignals()
      .then((data) => {
        cachedDefs = data.signals || [];
        return cachedDefs;
      })
      .catch(() => {
        loadPromise = null;
        return [] as TrainingSignalDef[];
      });
  }
  return loadPromise;
}

export function useTrainingSignalDefinitions() {
  const definitions = ref<TrainingSignalDef[]>(cachedDefs || []);
  const loading = ref(!cachedDefs);

  onMounted(async () => {
    if (cachedDefs) {
      definitions.value = cachedDefs;
      loading.value = false;
      return;
    }
    loading.value = true;
    definitions.value = await loadSignalDefinitions();
    loading.value = false;
  });

  return { definitions, loading };
}

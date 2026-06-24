import { ref, watch, type Ref } from "vue";
import { api } from "../api";
import {
  coerceTrainingDatasetEntries,
  libraryDatasetIdFromRef,
} from "../lib/datasetLibraryRef";
import { useLatestAsync } from "./useLatestAsync";
import type { FormValues } from "../types/forms";
import type { EstimateStepsResult } from "../types/api";

/**
 * Watches the training config form + dataset entries and fires a debounced
 * POST /api/v1/runs/estimate-steps whenever the relevant inputs change.
 *
 * `ok: false` responses are treated as "not available yet" (incomplete config),
 * not errors — the readout just shows a dash.
 *
 * @param form   - reactive ref to the config editor form (may be null while loading)
 * @param numGpus - reactive ref to the launch GPU count
 */
export function useEstimateSteps(
  form: Ref<FormValues | null>,
  numGpus: Ref<number>
) {
  const result = ref<EstimateStepsResult | null>(null);
  const loading = ref(false);

  const { begin, isCurrent, schedule, cancel } = useLatestAsync();

  /** Fetch TOML for a single library-ref dataset id. Returns null on failure. */
  async function fetchDatasetToml(id: string): Promise<string | null> {
    try {
      const data = (await api.getDataset(id)) as { content?: string };
      return typeof data.content === "string" && data.content.trim()
        ? data.content
        : null;
    } catch {
      return null;
    }
  }

  /**
   * Build a combined dataset TOML from the current dataset entries.
   * - Single library ref → fetch its TOML.
   * - Multiple refs → fetch each and concatenate (the server merges [[directory]] blocks).
   * - Raw paths → can't get TOML; omit them (server will return ok:false, shown as dash).
   * Returns an empty string when nothing can be resolved.
   */
  async function buildDatasetToml(entries: string[]): Promise<string> {
    const parts: string[] = [];
    for (const entry of entries) {
      const libId = libraryDatasetIdFromRef(entry);
      if (libId) {
        const toml = await fetchDatasetToml(libId);
        if (toml) parts.push(toml);
      }
      // Raw paths are omitted — endpoint will return ok:false, shown as dash.
    }
    return parts.join("\n");
  }

  /**
   * Extract the config fields the endpoint cares about from the form.
   * Passes only what the server needs; omits undefined fields so the server
   * can use its own defaults.
   */
  function buildConfigPayload(f: FormValues): Record<string, unknown> {
    const payload: Record<string, unknown> = {};
    if (f.epochs != null) payload.epochs = f.epochs;
    if (f.micro_batch_size_per_gpu != null)
      payload.micro_batch_size_per_gpu = f.micro_batch_size_per_gpu;
    if (f.gradient_accumulation_steps != null)
      payload.gradient_accumulation_steps = f.gradient_accumulation_steps;
    if (f.max_steps != null) payload.max_steps = f.max_steps;
    return payload;
  }

  async function fetchEstimate(): Promise<void> {
    const f = form.value;
    if (!f) return;

    const token = begin();
    loading.value = true;
    try {
      const entries = coerceTrainingDatasetEntries(f.dataset);
      const datasetToml = await buildDatasetToml(entries);

      // Guard: a stale call may have started fetchDatasetToml while a newer call
      // was already kicked off; bail if we're no longer the latest.
      if (!isCurrent(token)) return;

      const configPayload = buildConfigPayload(f);

      const res = await api.estimateSteps({
        dataset_toml: datasetToml,
        config: configPayload,
        num_gpus: numGpus.value,
      });

      if (!isCurrent(token)) return;

      if (res.ok) {
        result.value = res as EstimateStepsResult;
      } else {
        // ok:false = incomplete/invalid config — show dash, don't error-toast.
        result.value = null;
      }
    } catch {
      if (!isCurrent(token)) return;
      // Network failures / unexpected errors — silently show dash.
      result.value = null;
    } finally {
      if (isCurrent(token)) {
        loading.value = false;
      }
    }
  }

  /**
   * Derive a stable watch key from the form fields that affect step count.
   * Using a computed string avoids deep-watching the entire form object.
   */
  function watchKey(f: FormValues | null, gpus: number): string {
    if (!f) return "";
    return JSON.stringify({
      dataset: f.dataset,
      epochs: f.epochs,
      mbs: f.micro_batch_size_per_gpu,
      gas: f.gradient_accumulation_steps,
      max: f.max_steps,
      gpus,
    });
  }

  watch(
    () => watchKey(form.value, numGpus.value),
    (key, prev) => {
      if (key === prev) return;
      cancel();
      if (!key) {
        result.value = null;
        return;
      }
      schedule(() => void fetchEstimate(), 500);
    },
    { immediate: true }
  );

  return { result, loading };
}

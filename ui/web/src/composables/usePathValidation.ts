import { onBeforeUnmount, ref, type Ref } from "vue";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { PathExpect } from "../lib/pathFields";
import { shouldSkipPathValidation } from "../lib/pathFields";
import type { FsStatResult } from "../types/api";

export interface UsePathValidationOptions {
  debounceMs?: number;
  expect?: PathExpect | null | (() => PathExpect | null);
  required?: boolean | (() => boolean);
  skip?: (path: string) => boolean;
}

export interface UsePathValidationReturn {
  loading: Ref<boolean>;
  error: Ref<string>;
  ok: Ref<boolean>;
  validate: (path: string) => Promise<FsStatResult | null>;
  scheduleValidation: (path: string) => void;
  clear: () => void;
}

export function usePathValidation(
  options: UsePathValidationOptions = {}
): UsePathValidationReturn {
  const loading = ref(false);
  const error = ref("");
  const ok = ref(false);

  let timer: ReturnType<typeof setTimeout> | null = null;
  let seq = 0;

  function resolveExpect(): PathExpect | null {
    const exp = options.expect;
    if (typeof exp === "function") return exp();
    return exp ?? null;
  }

  function resolveRequired(): boolean {
    const req = options.required;
    if (typeof req === "function") return req();
    return !!req;
  }

  function shouldSkip(path: string): boolean {
    if (options.skip?.(path)) return true;
    return shouldSkipPathValidation(path, { required: resolveRequired() });
  }

  function clear(): void {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    loading.value = false;
    error.value = "";
    ok.value = false;
  }

  async function validate(path: string): Promise<FsStatResult | null> {
    const trimmed = (path || "").trim();
    if (shouldSkip(trimmed)) {
      clear();
      return null;
    }

    const mySeq = ++seq;
    loading.value = true;
    error.value = "";
    ok.value = false;

    try {
      const expect = resolveExpect();
      const result = await api.fsStat(trimmed, expect ?? undefined);
      if (mySeq !== seq) return result;

      if (result.error) {
        error.value = result.error;
        ok.value = false;
      } else if (result.exists) {
        error.value = "";
        ok.value = true;
      } else {
        error.value = "Path does not exist";
        ok.value = false;
      }
      return result;
    } catch (e) {
      if (mySeq !== seq) return null;
      error.value = formatError(e);
      ok.value = false;
      return null;
    } finally {
      if (mySeq === seq) {
        loading.value = false;
      }
    }
  }

  function scheduleValidation(path: string): void {
    if (timer) clearTimeout(timer);
    const trimmed = (path || "").trim();
    if (shouldSkip(trimmed)) {
      clear();
      return;
    }
    timer = setTimeout(() => {
      timer = null;
      void validate(trimmed);
    }, options.debounceMs ?? 450);
  }

  onBeforeUnmount(() => {
    if (timer) clearTimeout(timer);
    seq += 1;
  });

  return {
    loading,
    error,
    ok,
    validate,
    scheduleValidation,
    clear,
  };
}

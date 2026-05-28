import { computed, onMounted, onUnmounted, ref, watch, type Ref } from "vue";
import {
  AUTO_REFRESH_STORAGE_KEY,
  DEFAULT_AUTO_REFRESH_SEC,
  effectiveRefreshMs,
  parseStoredInterval,
  type AutoRefreshIntervalSec,
} from "../lib/autoRefresh";

export interface UseAutoRefreshOptions {
  refresh: (signal: AbortSignal) => Promise<void>;
  isActive?: () => boolean;
  storageKey?: string;
  /** Run once on mount before scheduling (default true). */
  immediate?: boolean;
}

export interface UseAutoRefreshReturn {
  intervalSec: Ref<AutoRefreshIntervalSec>;
  /** True while any refresh (manual or scheduled) is in flight. */
  isLoading: Ref<boolean>;
  /** True only during manual refresh (toolbar button). */
  refreshing: Ref<boolean>;
  /** True during scheduled auto-refresh (subtle toolbar hint). */
  polling: Ref<boolean>;
  lastUpdated: Ref<Date | null>;
  paused: Ref<boolean>;
  setIntervalSec: (sec: AutoRefreshIntervalSec) => void;
  refreshNow: () => Promise<void>;
}

export function useAutoRefresh(options: UseAutoRefreshOptions): UseAutoRefreshReturn {
  const storageKey = options.storageKey ?? AUTO_REFRESH_STORAGE_KEY;
  const intervalSec = ref<AutoRefreshIntervalSec>(
    parseStoredInterval(typeof localStorage !== "undefined" ? localStorage.getItem(storageKey) : null)
  );
  const isLoading = ref(false);
  const refreshing = ref(false);
  const polling = ref(false);
  const lastUpdated = ref<Date | null>(null);
  const paused = ref(typeof document !== "undefined" && document.hidden);

  let timer: ReturnType<typeof setTimeout> | null = null;
  let abortController: AbortController | null = null;
  let generation = 0;

  const activeFlag = computed(() => options.isActive?.() ?? false);

  const effectiveMs = computed(() =>
    effectiveRefreshMs(intervalSec.value, activeFlag.value)
  );

  function clearTimer(): void {
    if (timer) clearTimeout(timer);
    timer = null;
  }

  function abortInFlight(): void {
    abortController?.abort();
    abortController = null;
  }

  function restartTimer(): void {
    clearTimer();
    if (effectiveMs.value > 0 && !paused.value) {
      scheduleNext(effectiveMs.value);
    }
  }

  function scheduleNext(delayMs?: number): void {
    clearTimer();
    const ms = delayMs ?? effectiveMs.value;
    if (ms <= 0 || paused.value) return;
    timer = setTimeout(() => {
      void runRefresh(false);
    }, ms);
  }

  async function runRefresh(manual: boolean): Promise<void> {
    if (!manual && (paused.value || effectiveMs.value <= 0)) {
      scheduleNext();
      return;
    }

    if (manual) clearTimer();

    abortInFlight();
    const ac = new AbortController();
    abortController = ac;
    const gen = ++generation;
    isLoading.value = true;
    if (manual) refreshing.value = true;
    else polling.value = true;

    try {
      await options.refresh(ac.signal);
      if (!ac.signal.aborted && gen === generation) {
        lastUpdated.value = new Date();
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      throw e;
    } finally {
      if (gen === generation) {
        isLoading.value = false;
        refreshing.value = false;
        polling.value = false;
        abortController = null;
        scheduleNext();
      }
    }
  }

  function refreshNow(): Promise<void> {
    return runRefresh(true);
  }

  function setIntervalSec(sec: AutoRefreshIntervalSec): void {
    intervalSec.value = sec;
    try {
      localStorage.setItem(storageKey, String(sec));
    } catch {
      /* private mode */
    }
    clearTimer();
    abortInFlight();
    isLoading.value = false;
    refreshing.value = false;
    polling.value = false;
    if (sec > 0 && !paused.value) {
      void runRefresh(true);
    }
  }

  function onVisibilityChange(): void {
    paused.value = document.hidden;
    if (document.hidden) {
      clearTimer();
      abortInFlight();
      isLoading.value = false;
      refreshing.value = false;
      polling.value = false;
    } else if (effectiveMs.value > 0) {
      void runRefresh(true);
    }
  }

  watch(intervalSec, restartTimer);
  watch(activeFlag, restartTimer);
  watch(effectiveMs, (ms, prev) => {
    if (ms === prev) return;
    restartTimer();
  });

  onMounted(() => {
    document.addEventListener("visibilitychange", onVisibilityChange);
    if (options.immediate !== false) {
      void runRefresh(true);
    }
  });

  onUnmounted(() => {
    document.removeEventListener("visibilitychange", onVisibilityChange);
    clearTimer();
    abortInFlight();
    generation++;
  });

  return {
    intervalSec,
    isLoading,
    refreshing,
    polling,
    lastUpdated,
    paused,
    setIntervalSec,
    refreshNow,
  };
}

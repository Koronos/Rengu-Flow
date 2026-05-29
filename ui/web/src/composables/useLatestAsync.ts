import { onUnmounted } from "vue";

/**
 * Latest-wins async guard for debounced loaders.
 *
 * Encapsulates the generation-counter + debounce-timer + unmount-cleanup
 * pattern that several composables repeat: only the most recent attempt may
 * apply its results, any pending debounce is replaced, and both are cancelled
 * when the owning component unmounts (avoiding stale state writes / leaks).
 */
export function useLatestAsync() {
  let generation = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  /** Start a new attempt and return its token. */
  function begin(): number {
    return ++generation;
  }

  /** True while `token` is the latest attempt — i.e. its results may be applied. */
  function isCurrent(token: number): boolean {
    return token === generation;
  }

  /** Debounce `fn` by `ms`, replacing any pending call. */
  function schedule(fn: () => void, ms: number): void {
    if (timer) clearTimeout(timer);
    timer = setTimeout(fn, ms);
  }

  /** Invalidate any in-flight attempt and cancel a pending debounce. */
  function cancel(): void {
    generation += 1;
    if (timer) clearTimeout(timer);
    timer = undefined;
  }

  onUnmounted(cancel);

  return { begin, isCurrent, schedule, cancel };
}

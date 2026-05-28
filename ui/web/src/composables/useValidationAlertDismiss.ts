/** Auto-dismiss validation feedback bars in config/dataset editors. */
export const VALIDATION_ALERT_ERROR_MS = 10_000;
export const VALIDATION_ALERT_SUCCESS_MS = 5_000;

export function createValidationAlertScheduler() {
  let errorTimer: ReturnType<typeof setTimeout> | undefined;
  let successTimer: ReturnType<typeof setTimeout> | undefined;

  function clearErrorTimer() {
    if (errorTimer !== undefined) {
      clearTimeout(errorTimer);
      errorTimer = undefined;
    }
  }

  function clearSuccessTimer() {
    if (successTimer !== undefined) {
      clearTimeout(successTimer);
      successTimer = undefined;
    }
  }

  function clearAll() {
    clearErrorTimer();
    clearSuccessTimer();
  }

  function scheduleErrorDismiss(clear: () => void) {
    clearErrorTimer();
    errorTimer = setTimeout(() => {
      errorTimer = undefined;
      clear();
    }, VALIDATION_ALERT_ERROR_MS);
  }

  function scheduleSuccessDismiss(clear: () => void) {
    clearSuccessTimer();
    successTimer = setTimeout(() => {
      successTimer = undefined;
      clear();
    }, VALIDATION_ALERT_SUCCESS_MS);
  }

  return {
    scheduleErrorDismiss,
    scheduleSuccessDismiss,
    clearAll,
  };
}

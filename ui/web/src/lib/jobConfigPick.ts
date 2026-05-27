/** Persist which library config is selected for the next training job. */

const STORAGE_KEY = "renga_flow_job_config_id";

export function getJobConfigId() {
  try {
    return sessionStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setJobConfigId(id) {
  try {
    if (id) sessionStorage.setItem(STORAGE_KEY, id);
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* private mode */
  }
}

import type { Router } from "vue-router";

/** Redirect to a stored job config when pick-for-job flow resumes from Runs. */
export async function redirectToStoredJobConfig(
  router: Router,
  storedConfigId: string | null | undefined,
  verifyExists: (id: string) => Promise<unknown>
): Promise<boolean> {
  const id = storedConfigId?.trim();
  if (!id) return false;
  try {
    await verifyExists(id);
    await router.replace({
      name: "configs-detail",
      params: { configId: id },
      query: { pick: "job" },
    });
    return true;
  } catch {
    return false;
  }
}

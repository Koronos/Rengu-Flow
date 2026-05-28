import { useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { ImportConfigResult } from "../types/api";

/** Import a config TOML file and navigate to the new config editor. */
export function useImportConfigToml(options?: { onError?: (msg: string) => void }) {
  const router = useRouter();

  async function importConfigFile(file: File): Promise<void> {
    try {
      const text = await file.text();
      const base = file.name.replace(/\.toml$/i, "") || "imported";
      const r = (await api.importConfig(text, base)) as ImportConfigResult;
      ElMessage.success(`Imported as ${r.id}`);
      await router.push({ name: "configs-detail", params: { configId: String(r.id) } });
    } catch (e) {
      const msg = formatError(e);
      options?.onError?.(msg);
      ElMessage.error(msg);
    }
  }

  return { importConfigFile };
}

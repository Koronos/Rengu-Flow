import { ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { TensorboardStartResult, TensorboardStatus } from "../types/api";

export function useTensorboard(getOutputDir: () => string | undefined) {
  const tbLoading = ref(false);
  const tbStatus = ref<TensorboardStatus | null>(null);

  async function refreshTbStatus() {
    try {
      tbStatus.value = (await api.tensorboardStatus()) as TensorboardStatus;
    } catch {
      tbStatus.value = null;
    }
  }

  async function openTensorboard({ onError }: { onError?: (msg: string) => void } = {}) {
    const outputDir = getOutputDir();
    tbLoading.value = true;
    try {
      const r = (await api.tensorboardStart({
        output_dir: outputDir || "output",
      })) as TensorboardStartResult & { reused?: boolean };
      tbStatus.value = r;
      if (r.url) window.open(String(r.url), "_blank", "noopener,noreferrer");
      ElMessage.success(r.reused ? "TensorBoard already running" : "TensorBoard started");
    } catch (e) {
      const msg = formatError(e);
      onError?.(msg);
      ElMessage.error(msg);
      throw e;
    } finally {
      tbLoading.value = false;
    }
  }

  async function stopTensorboard({ onError }: { onError?: (msg: string) => void } = {}) {
    tbLoading.value = true;
    try {
      await api.tensorboardStop();
      tbStatus.value = { running: false };
      ElMessage.success("TensorBoard stopped");
    } catch (e) {
      const msg = formatError(e);
      onError?.(msg);
      ElMessage.error(msg);
      throw e;
    } finally {
      tbLoading.value = false;
    }
  }

  return { tbLoading, tbStatus, refreshTbStatus, openTensorboard, stopTensorboard };
}

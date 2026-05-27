import { ref } from "vue";
import { ElMessage } from "element-plus";
import { api } from "../api";

export function useTensorboard(getOutputDir) {
  const tbLoading = ref(false);
  const tbStatus = ref(null);

  async function refreshTbStatus() {
    try {
      tbStatus.value = await api.tensorboardStatus();
    } catch {
      tbStatus.value = null;
    }
  }

  async function openTensorboard({ onError } = {}) {
    const outputDir = typeof getOutputDir === "function" ? getOutputDir() : getOutputDir;
    tbLoading.value = true;
    try {
      const r = await api.tensorboardStart({ output_dir: outputDir || "output" });
      tbStatus.value = r;
      if (r.url) window.open(r.url, "_blank", "noopener,noreferrer");
      ElMessage.success(r.reused ? "TensorBoard already running" : "TensorBoard started");
    } catch (e) {
      const msg = String(e);
      if (onError) onError(msg);
      ElMessage.error(msg);
      throw e;
    } finally {
      tbLoading.value = false;
    }
  }

  async function stopTensorboard({ onError } = {}) {
    tbLoading.value = true;
    try {
      await api.tensorboardStop();
      tbStatus.value = { running: false };
      ElMessage.success("TensorBoard stopped");
    } catch (e) {
      const msg = String(e);
      if (onError) onError(msg);
      ElMessage.error(msg);
      throw e;
    } finally {
      tbLoading.value = false;
    }
  }

  return { tbLoading, tbStatus, refreshTbStatus, openTensorboard, stopTensorboard };
}

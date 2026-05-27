import { formatApiDetail } from "./lib/formatError";
import { filenameFromContentDisposition } from "./lib/downloadBlob";
import type { FormValues } from "./types/forms";

const API = "/api/v1";

async function request<T = unknown>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const body = data as Record<string, unknown> | null;
    const msg =
      formatApiDetail(body?.detail) ||
      (typeof body?.error === "string" ? body.error : formatApiDetail(body?.error)) ||
      res.statusText;
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return data as T;
}

export const api = {
  listConfigs: () => request("/configs"),
  searchConfigs: (params) => {
    const q = params instanceof URLSearchParams ? params : new URLSearchParams(params);
    if (!q.has("page")) q.set("page", "1");
    if (!q.has("page_size")) q.set("page_size", "20");
    return request(`/configs?${q.toString()}`);
  },
  getConfig: (id) => request(`/configs/${id}`),
  saveConfig: (id, content) =>
    request(`/configs/${id}`, { method: "PUT", body: JSON.stringify({ content }) }),
  createConfig: (id, content) =>
    request("/configs", { method: "POST", body: JSON.stringify({ id, content }) }),
  deleteConfig: (id) => request(`/configs/${id}`, { method: "DELETE" }),
  validate: (content) =>
    request("/validate", { method: "POST", body: JSON.stringify({ content }) }),
  duplicate: (id) => request(`/configs/${id}/duplicate`, { method: "POST" }),
  importExample: (path, configId) =>
    request(
      `/configs/import-example?path=${encodeURIComponent(path)}${configId ? `&config_id=${encodeURIComponent(configId)}` : ""}`,
      { method: "POST" }
    ),
  importConfig: (content, id) =>
    request("/configs/import", { method: "POST", body: JSON.stringify({ id, content }) }),
  async exportConfigBundle(configId, content) {
    const hasInline = typeof content === "string";
    const url = hasInline
      ? `${API}/configs/export-bundle`
      : `${API}/configs/${encodeURIComponent(configId)}/export`;
    const res = await fetch(url, {
      method: hasInline ? "POST" : "GET",
      headers: hasInline ? { "Content-Type": "application/json" } : {},
      body: hasInline
        ? JSON.stringify({ content, name: configId || "training_export" })
        : undefined,
    });
    const text = await res.text();
    if (!res.ok) {
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = { detail: text };
      }
      const msg =
        formatApiDetail(data?.detail) ||
        (typeof data?.error === "string" ? data.error : formatApiDetail(data?.error)) ||
        res.statusText;
      throw new Error(msg || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const filename =
      filenameFromContentDisposition(res.headers.get("Content-Disposition")) ||
      `${configId || "training_export"}.zip`;
    return { blob, filename };
  },
  listJobs: () => request("/jobs"),
  trainRuns: (params) => {
    const q = params instanceof URLSearchParams ? params : new URLSearchParams(params);
    if (!q.has("page")) q.set("page", "1");
    if (!q.has("page_size")) q.set("page_size", "20");
    return request(`/train/runs?${q.toString()}`);
  },
  trainActive: () => request("/train/active"),
  listImportCandidates: (outputDir = "output") =>
    request(`/jobs/import/candidates?output_dir=${encodeURIComponent(outputDir)}`),
  previewJobImport: (runPath) =>
    request("/jobs/import/preview", {
      method: "POST",
      body: JSON.stringify({ run_path: runPath }),
    }),
  importJobFromRun: (body) =>
    request("/jobs/import", { method: "POST", body: JSON.stringify(body) }),
  getRunConfig: (runPath) =>
    request(`/runs/config?run_path=${encodeURIComponent(runPath)}`),
  continueRun: (body) =>
    request("/jobs/continue-run", { method: "POST", body: JSON.stringify(body) }),
  getJob: (id) => request(`/jobs/${id}`),
  jobArtifacts: (id) => request(`/jobs/${id}/artifacts`),
  startJob: (body) => request("/jobs", { method: "POST", body: JSON.stringify(body) }),
  updateJob: (id, body) =>
    request(`/jobs/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteJob: (id) => request(`/jobs/${id}`, { method: "DELETE" }),
  moveJobQueue: (id, direction) =>
    request(`/jobs/${id}/queue/move?direction=${direction}`, { method: "POST" }),
  startJobNow: (id) => request(`/jobs/${id}/queue/start-now`, { method: "POST" }),
  stopJob: (id) => request(`/jobs/${id}/stop`, { method: "POST" }),
  sendJobSignal: (id, type) =>
    request(`/jobs/${id}/signals`, { method: "POST", body: JSON.stringify({ type }) }),
  jobMetrics: (id) => request(`/jobs/${id}/metrics`),
  listFsRuns: (outputDir = "output") =>
    request(`/runs?output_dir=${encodeURIComponent(outputDir)}`),
  getFsRun: (name, outputDir = "output") =>
    request(`/runs/${encodeURIComponent(name)}?output_dir=${encodeURIComponent(outputDir)}`),
  fsSignal: (name, type, outputDir = "output") =>
    request(
      `/runs/${encodeURIComponent(name)}/signals?output_dir=${encodeURIComponent(outputDir)}`,
      { method: "POST", body: JSON.stringify({ type }) }
    ),
  fsMetrics: (name, outputDir = "output") =>
    request(`/runs/${encodeURIComponent(name)}/metrics?output_dir=${encodeURIComponent(outputDir)}`),
  getSchema: () => request("/schema"),
  getSystemStats: () => request("/system/stats"),
  tensorboardStatus: () => request("/tensorboard/status"),
  tensorboardStart: (body = {}) =>
    request("/tensorboard/start", { method: "POST", body: JSON.stringify(body) }),
  tensorboardStop: () => request("/tensorboard/stop", { method: "POST" }),
  probeRegistry: (body) =>
    request("/registry/probe", { method: "POST", body: JSON.stringify(body) }),
  getDoc: (path) => request(`/docs?path=${encodeURIComponent(path)}`),
  getDocsIndex: () => request("/docs/index"),
  parseToml: (content) =>
    request("/configs/parse-toml", { method: "POST", body: JSON.stringify({ content }) }),
  renderToml: (form: FormValues) => {
    if (!form || typeof form !== "object" || Array.isArray(form)) {
      return Promise.reject(new Error("Config form is not ready yet."));
    }
    return request("/configs/render-toml", {
      method: "POST",
      body: JSON.stringify({ form }),
    });
  },

  listDatasets: () => request("/datasets"),
  searchDatasets: (params) => {
    const q = params instanceof URLSearchParams ? params : new URLSearchParams(params);
    if (!q.has("page")) q.set("page", "1");
    if (!q.has("page_size")) q.set("page_size", "20");
    return request(`/datasets?${q.toString()}`);
  },
  getDataset: (id) => request(`/datasets/${id}`),
  saveDataset: (id, payload) =>
    request(`/datasets/${id}`, {
      method: "PUT",
      body: JSON.stringify(
        typeof payload === "string" ? { content: payload } : payload
      ),
    }),
  createDataset: (payload) => {
    const body =
      typeof payload === "string" ? { content: payload } : { ...payload };
    return request("/datasets", { method: "POST", body: JSON.stringify(body) });
  },
  deleteDataset: (id) => request(`/datasets/${id}`, { method: "DELETE" }),
  duplicateDataset: (id, newId) =>
    request(
      `/datasets/${id}/duplicate${newId ? `?new_id=${encodeURIComponent(newId)}` : ""}`,
      { method: "POST" }
    ),
  validateDataset: (content) =>
    request("/datasets/validate", { method: "POST", body: JSON.stringify({ content }) }),
  importDatasetExample: (path, datasetId) =>
    request(
      `/datasets/import-example?path=${encodeURIComponent(path)}${datasetId ? `&dataset_id=${encodeURIComponent(datasetId)}` : ""}`,
      { method: "POST" }
    ),
  importDataset: (content: string, id?: string) => {
    const body: { content: string; id?: string } = { content };
    if (id) body.id = id;
    return request("/datasets/import", { method: "POST", body: JSON.stringify(body) });
  },
  exportDataset: (id) => request(`/datasets/${encodeURIComponent(id)}/export`),
  getDatasetSchema: () => request("/datasets/schema"),
  getDatasetFolderSuggestions: (excludeId) => {
    const q = excludeId ? `?exclude=${encodeURIComponent(excludeId)}` : "";
    return request(`/datasets/folder-suggestions${q}`);
  },
  parseDatasetToml: (content) =>
    request("/datasets/parse-toml", { method: "POST", body: JSON.stringify({ content }) }),
  renderDatasetToml: (form: FormValues) => {
    if (!form || typeof form !== "object" || Array.isArray(form)) {
      return Promise.reject(new Error("Dataset form is not ready yet."));
    }
    return request("/datasets/render-toml", {
      method: "POST",
      body: JSON.stringify({ form }),
    });
  },
  previewDataset: (content) =>
    request("/datasets/preview", { method: "POST", body: JSON.stringify({ content }) }),
  listDatasetPreviewImages: (body) =>
    request("/datasets/preview-images", { method: "POST", body: JSON.stringify(body) }),
  datasetPreviewImageUrl: (token) =>
    `${API}/datasets/preview-image?t=${encodeURIComponent(token)}`,
  scanDatasetPath: (path) =>
    request("/datasets/scan-path", { method: "POST", body: JSON.stringify({ path }) }),
  composeDatasets: (sourceIds: string[], targetId?: string) => {
    const body: { source_ids: string[]; target_id?: string } = { source_ids: sourceIds };
    if (targetId) body.target_id = targetId;
    return request("/datasets/compose", { method: "POST", body: JSON.stringify(body) });
  },
};

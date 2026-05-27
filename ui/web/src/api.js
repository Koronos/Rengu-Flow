const API = "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    throw new Error(data?.detail || data?.error || res.statusText);
  }
  return data;
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
  exportConfig: (id) => request(`/configs/${encodeURIComponent(id)}/export`),
  listJobs: () => request("/jobs"),
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
  parseToml: (content) =>
    request("/configs/parse-toml", { method: "POST", body: JSON.stringify({ content }) }),
  renderToml: (form) =>
    request("/configs/render-toml", { method: "POST", body: JSON.stringify({ form }) }),

  listDatasets: () => request("/datasets"),
  searchDatasets: (params) => {
    const q = params instanceof URLSearchParams ? params : new URLSearchParams(params);
    if (!q.has("page")) q.set("page", "1");
    if (!q.has("page_size")) q.set("page_size", "20");
    return request(`/datasets?${q.toString()}`);
  },
  getDataset: (id) => request(`/datasets/${id}`),
  saveDataset: (id, content) =>
    request(`/datasets/${id}`, { method: "PUT", body: JSON.stringify({ content }) }),
  createDataset: (id, content) =>
    request("/datasets", { method: "POST", body: JSON.stringify({ id, content }) }),
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
  importDataset: (content, id) =>
    request("/datasets/import", { method: "POST", body: JSON.stringify({ id, content }) }),
  exportDataset: (id) => request(`/datasets/${encodeURIComponent(id)}/export`),
  getDatasetSchema: () => request("/datasets/schema"),
  parseDatasetToml: (content) =>
    request("/datasets/parse-toml", { method: "POST", body: JSON.stringify({ content }) }),
  renderDatasetToml: (form) =>
    request("/datasets/render-toml", { method: "POST", body: JSON.stringify({ form }) }),
  previewDataset: (content) =>
    request("/datasets/preview", { method: "POST", body: JSON.stringify({ content }) }),
  listDatasetPreviewImages: (body) =>
    request("/datasets/preview-images", { method: "POST", body: JSON.stringify(body) }),
  datasetPreviewImageUrl: (token) =>
    `${API}/datasets/preview-image?t=${encodeURIComponent(token)}`,
  scanDatasetPath: (path) =>
    request("/datasets/scan-path", { method: "POST", body: JSON.stringify({ path }) }),
  composeDatasets: (targetId, sourceIds) =>
    request("/datasets/compose", {
      method: "POST",
      body: JSON.stringify({ target_id: targetId, source_ids: sourceIds }),
    }),
};

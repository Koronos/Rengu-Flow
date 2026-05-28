import { errorMessageFromResponseBody } from "./lib/formatError";
import { filenameFromContentDisposition } from "./lib/downloadBlob";
import type { FormValues } from "./types/forms";
import type {
  ConfigDetail,
  ConfigSchemaResponse,
  ConfigSearchItem,
  ContinueRunBody,
  DatasetComposeResult,
  DatasetDetail,
  DatasetFolderSuggestion,
  DatasetPreviewImagesResult,
  DatasetPreviewResult,
  DatasetSavePayload,
  DatasetScanPathResult,
  DatasetSchemaResponse,
  DatasetSearchItem,
  DocContentResult,
  DocsIndexResult,
  DuplicateConfigResult,
  ExportBundleResult,
  FsRunRecord,
  FsRunsListResult,
  FsStatResult,
  ImportCandidatesResult,
  ImportConfigResult,
  ImportExampleResult,
  ImportRunPreview,
  JobImportBody,
  JobListResult,
  JobMetricsResult,
  JobPatchBody,
  JobRecord,
  JobStartBody,
  Paginated,
  ParseTomlResult,
  QueryParams,
  RegistryProbeBody,
  RegistryProbeResult,
  RenderTomlResult,
  RunConfigResult,
  SystemStatsResponse,
  TensorboardStartBody,
  TensorboardStartResult,
  TensorboardStatus,
  TrainActiveResult,
  TrainingRunRow,
  TrainingSignalsResult,
  TrainRunsResult,
  ValidateResult,
  AugmentationCatalogResponse,
  MaintenanceCommandOutput,
  MaintenanceDbResetResult,
  MaintenanceEnabledResult,
  MaintenanceStatus,
} from "./types/api";
import { withDefaultPagination } from "./types/api";

const API = "/api/v1";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
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
    const msg = errorMessageFromResponseBody(data, res.statusText);
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return data as T;
}

export const api = {
  listConfigs: () => request<ConfigSearchItem[]>("/configs"),

  searchConfigs: (params: QueryParams) => {
    const q = withDefaultPagination(params, { page: "1", page_size: "20" });
    return request<Paginated<ConfigSearchItem>>(`/configs?${q.toString()}`);
  },

  getConfig: (id: number | string) => request<ConfigDetail>(`/configs/${id}`),

  saveConfig: (id: number | string, content: string) =>
    request<ConfigDetail>(`/configs/${id}`, {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),

  createConfig: (id: number | string, content: string) =>
    request<ConfigDetail>("/configs", {
      method: "POST",
      body: JSON.stringify({ id, content }),
    }),

  deleteConfig: (id: number | string) =>
    request<void>(`/configs/${id}`, { method: "DELETE" }),

  validate: (content: string) =>
    request<ValidateResult>("/validate", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  duplicate: (id: number | string) =>
    request<DuplicateConfigResult>(`/configs/${id}/duplicate`, { method: "POST" }),

  importConfig: (content: string, id?: number | string) =>
    request<ImportConfigResult>("/configs/import", {
      method: "POST",
      body: JSON.stringify({ id, content }),
    }),

  async exportConfigBundle(
    configId: number | string,
    content?: string
  ): Promise<ExportBundleResult> {
    const hasInline = typeof content === "string";
    const url = hasInline
      ? `${API}/configs/export-bundle`
      : `${API}/configs/${encodeURIComponent(String(configId))}/export`;
    const res = await fetch(url, {
      method: hasInline ? "POST" : "GET",
      headers: hasInline ? { "Content-Type": "application/json" } : {},
      body: hasInline
        ? JSON.stringify({ content, name: configId || "training_export" })
        : undefined,
    });
    if (!res.ok) {
      const text = await res.text();
      let data: unknown = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = { detail: text };
      }
      const msg = errorMessageFromResponseBody(data, res.statusText);
      throw new Error(msg || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const filename =
      filenameFromContentDisposition(res.headers.get("Content-Disposition")) ||
      `${configId || "training_export"}.zip`;
    return { blob, filename };
  },

  listJobs: () => request<JobListResult>("/jobs"),

  trainRuns: (params: QueryParams) => {
    const q = withDefaultPagination(params, { page: "1", page_size: "20" });
    return request<TrainRunsResult>(`/train/runs?${q.toString()}`);
  },

  trainActive: () => request<TrainActiveResult>("/train/active"),

  listImportCandidates: (outputDir = "output") =>
    request<ImportCandidatesResult>(
      `/jobs/import/candidates?output_dir=${encodeURIComponent(outputDir)}`
    ),

  previewJobImport: (runPath: string) =>
    request<ImportRunPreview>("/jobs/import/preview", {
      method: "POST",
      body: JSON.stringify({ run_path: runPath }),
    }),

  importJobFromRun: (body: JobImportBody) =>
    request<JobRecord>("/jobs/import", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getRunConfig: (runPath: string) =>
    request<RunConfigResult>(`/runs/config?run_path=${encodeURIComponent(runPath)}`),

  continueRun: (body: ContinueRunBody) =>
    request<JobRecord>("/jobs/continue-run", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getJob: (id: string) => request<JobRecord>(`/jobs/${id}`),

  jobArtifacts: (id: string) =>
    request<{ artifacts?: Record<string, unknown>[] }>(`/jobs/${id}/artifacts`),

  startJob: (body: JobStartBody) =>
    request<JobRecord>("/jobs", { method: "POST", body: JSON.stringify(body) }),

  updateJob: (id: string, body: JobPatchBody) =>
    request<JobRecord>(`/jobs/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: "DELETE" }),

  moveJobQueue: (id: string, direction: "up" | "down") =>
    request<JobRecord>(`/jobs/${id}/queue/move?direction=${direction}`, {
      method: "POST",
    }),

  startJobNow: (id: string) =>
    request<JobRecord>(`/jobs/${id}/queue/start-now`, { method: "POST" }),

  stopJob: (id: string) => request<JobRecord>(`/jobs/${id}/stop`, { method: "POST" }),

  sendJobSignal: (id: string, type: string) =>
    request<void>(`/jobs/${id}/signals`, {
      method: "POST",
      body: JSON.stringify({ type }),
    }),

  listSignals: () => request<TrainingSignalsResult>("/signals"),

  jobMetrics: (id: string) => request<JobMetricsResult>(`/jobs/${id}/metrics`),

  jobLogs: (id: string, offset = 0) =>
    request<{ chunk: string; offset: number }>(`/jobs/${id}/logs?offset=${offset}`),

  listFsRuns: (outputDir = "output") =>
    request<FsRunsListResult>(`/runs?output_dir=${encodeURIComponent(outputDir)}`),

  getFsRun: (name: string, outputDir = "output") =>
    request<FsRunRecord>(
      `/runs/${encodeURIComponent(name)}?output_dir=${encodeURIComponent(outputDir)}`
    ),

  fsSignal: (name: string, type: string, outputDir = "output") =>
    request<void>(
      `/runs/${encodeURIComponent(name)}/signals?output_dir=${encodeURIComponent(outputDir)}`,
      { method: "POST", body: JSON.stringify({ type }) }
    ),

  fsMetrics: (name: string, outputDir = "output") =>
    request<JobMetricsResult>(
      `/runs/${encodeURIComponent(name)}/metrics?output_dir=${encodeURIComponent(outputDir)}`
    ),

  getSchema: () => request<ConfigSchemaResponse>("/schema"),

  getSystemStats: () => request<SystemStatsResponse>("/system/stats"),

  tensorboardStatus: () => request<TensorboardStatus>("/tensorboard/status"),

  tensorboardStart: (body: TensorboardStartBody = {}) =>
    request<TensorboardStartResult>("/tensorboard/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  tensorboardStop: () => request<TensorboardStatus>("/tensorboard/stop", { method: "POST" }),

  probeRegistry: (body: RegistryProbeBody) =>
    request<RegistryProbeResult>("/registry/probe", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  getDoc: (path: string) =>
    request<DocContentResult>(`/docs?path=${encodeURIComponent(path)}`),

  getDocsIndex: () => request<DocsIndexResult>("/docs/index"),

  parseToml: (content: string) =>
    request<ParseTomlResult>("/configs/parse-toml", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  renderToml: (form: FormValues, baseContent?: string) => {
    if (!form || typeof form !== "object" || Array.isArray(form)) {
      return Promise.reject(new Error("Config form is not ready yet."));
    }
    const body: { form: FormValues; base_content?: string } = { form };
    if (baseContent?.trim()) {
      body.base_content = baseContent;
    }
    return request<RenderTomlResult>("/configs/render-toml", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  listDatasets: () => request<DatasetSearchItem[]>("/datasets"),

  searchDatasets: (params: QueryParams) => {
    const q = withDefaultPagination(params, { page: "1", page_size: "20" });
    return request<Paginated<DatasetSearchItem>>(`/datasets?${q.toString()}`);
  },

  getDataset: (id: number | string) => request<DatasetDetail>(`/datasets/${id}`),

  saveDataset: (id: number | string, payload: string | DatasetSavePayload) =>
    request<DatasetDetail>(`/datasets/${id}`, {
      method: "PUT",
      body: JSON.stringify(
        typeof payload === "string" ? { content: payload } : payload
      ),
    }),

  createDataset: (payload: string | DatasetSavePayload) => {
    const body =
      typeof payload === "string" ? { content: payload } : { ...payload };
    return request<DatasetDetail>("/datasets", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  deleteDataset: (id: number | string) =>
    request<void>(`/datasets/${id}`, { method: "DELETE" }),

  duplicateDataset: (id: number | string, newId?: string) =>
    request<DuplicateConfigResult>(
      `/datasets/${id}/duplicate${newId ? `?new_id=${encodeURIComponent(newId)}` : ""}`,
      { method: "POST" }
    ),

  validateDataset: (content: string) =>
    request<ValidateResult>("/datasets/validate", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  importDatasetExample: (path: string, datasetId?: number | string) =>
    request<ImportExampleResult>(
      `/datasets/import-example?path=${encodeURIComponent(path)}${
        datasetId ? `&dataset_id=${encodeURIComponent(String(datasetId))}` : ""
      }`,
      { method: "POST" }
    ),

  importDataset: (content: string, id?: string) => {
    const body: { content: string; id?: string } = { content };
    if (id) body.id = id;
    return request<ImportConfigResult>("/datasets/import", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  exportDataset: (id: number | string) =>
    request<{ content?: string }>(`/datasets/${encodeURIComponent(String(id))}/export`),

  getDatasetSchema: () => request<DatasetSchemaResponse>("/datasets/schema"),

  getAugmentationCatalog: () => request<AugmentationCatalogResponse>("/augmentations"),

  getDatasetFolderSuggestions: (excludeId?: number | string) => {
    const q = excludeId ? `?exclude=${encodeURIComponent(String(excludeId))}` : "";
    return request<{ suggestions?: DatasetFolderSuggestion[] }>(
      `/datasets/folder-suggestions${q}`
    );
  },

  parseDatasetToml: (content: string) =>
    request<ParseTomlResult>("/datasets/parse-toml", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  renderDatasetToml: (form: FormValues) => {
    if (!form || typeof form !== "object" || Array.isArray(form)) {
      return Promise.reject(new Error("Dataset form is not ready yet."));
    }
    return request<RenderTomlResult>("/datasets/render-toml", {
      method: "POST",
      body: JSON.stringify({ form }),
    });
  },

  previewDataset: (content: string) =>
    request<DatasetPreviewResult>("/datasets/preview", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  listDatasetPreviewImages: (body: {
    content: string;
    directory_index?: number;
    limit?: number;
    offset?: number;
  }) =>
    request<DatasetPreviewImagesResult>("/datasets/preview-images", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  datasetPreviewImageUrl: (token: string) =>
    `${API}/datasets/preview-image?t=${encodeURIComponent(token)}`,

  scanDatasetPath: (path: string) =>
    request<DatasetScanPathResult>("/datasets/scan-path", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  fsStat: (path: string, expect?: "file" | "dir") =>
    request<FsStatResult>("/fs/stat", {
      method: "POST",
      body: JSON.stringify({ path, expect: expect ?? null }),
    }),

  composeDatasets: (sourceIds: string[], targetId?: string) => {
    const body: { source_ids: string[]; target_id?: string } = { source_ids: sourceIds };
    if (targetId) body.target_id = targetId;
    return request<DatasetComposeResult>("/datasets/compose", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  maintenanceEnabled: () => request<MaintenanceEnabledResult>("/maintenance/enabled"),

  maintenanceStatus: () => request<MaintenanceStatus>("/maintenance/status"),

  maintenanceDatabaseReset: () =>
    request<MaintenanceDbResetResult>("/maintenance/database/reset", {
      method: "POST",
      body: JSON.stringify({ confirmation: "RESET" }),
    }),

  maintenanceSubmodulesUpdate: () =>
    request<MaintenanceCommandOutput>("/maintenance/submodules/update", {
      method: "POST",
      body: JSON.stringify({}),
    }),

  maintenanceDepsInstall: (profile: string, execute: boolean) =>
    request<MaintenanceCommandOutput>("/maintenance/deps/install", {
      method: "POST",
      body: JSON.stringify({ profile, execute, confirm: execute }),
    }),
};

import { errorMessageFromResponseBody } from "./lib/formatError";
import { filenameFromContentDisposition } from "./lib/downloadBlob";
import type { FormValues } from "./types/forms";
import type {
  CheckpointsResult,
  ConfigSchemaResponse,
  ContinueRunBody,
  CloneJobBody,
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
  ValidateOnlyResult,
  ValidateResult,
  AugmentationCatalogResponse,
  MaintenanceCommandOutput,
  MaintenanceDbResetResult,
  MaintenanceEnabledResult,
  MaintenanceStatus,
  VersionInfo,
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
  /** renga version + git commit + installed koptim, for the sidebar version label. */
  version: () => request<VersionInfo>("/version"),

  validate: (content: string) =>
    request<ValidateResult>("/validate", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  /** Full CLI pre-flight validation (rengu_flow.main --validate-only). Slower; runs a subprocess. */
  validateOnly: (content: string) =>
    request<ValidateOnlyResult>("/validate-only", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  /** Export the run's config + resolved dataset TOMLs as a ZIP for CLI training. */
  async exportConfigBundle(
    name: string,
    content: string
  ): Promise<ExportBundleResult> {
    const res = await fetch(`${API}/configs/export-bundle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, name: name || "training_export" }),
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
      `${name || "training_export"}.zip`;
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

  cloneJob: (id: string, body: CloneJobBody = {}) =>
    request<JobRecord>(`/jobs/${id}/clone`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  updateJob: (id: string, body: JobPatchBody) =>
    request<JobRecord>(`/jobs/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: "DELETE" }),

  /** Config TOML for a new run seeded from an existing one (run_name gets a _N suffix). */
  seedJobConfig: (id: string) => request<{ content: string }>(`/jobs/${id}/seed`),

  /** Promote a saved (new) draft into the pending queue. */
  enqueueJob: (id: string) =>
    request<JobRecord>(`/jobs/${id}/enqueue`, { method: "POST" }),

  /** Remove a queued (pending) run from the queue, keeping it as a saved draft. */
  dequeueJob: (id: string) =>
    request<JobRecord>(`/jobs/${id}/dequeue`, { method: "POST" }),

  /** Set the pending-queue order from an explicit list of job ids. */
  reorderQueue: (ids: (string | number)[]) =>
    request<{ queue: JobRecord[] }>("/jobs/queue/reorder", {
      method: "POST",
      body: JSON.stringify({ ids: ids.map((x) => Number(x)) }),
    }),

  jobCheckpoints: (id: string) =>
    request<CheckpointsResult>(`/jobs/${id}/checkpoints`),

  runCheckpoints: (runDir: string) =>
    request<CheckpointsResult>(`/runs/checkpoints?run_dir=${encodeURIComponent(runDir)}`),

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

  /** Current [preview] table for a running job's live config. */
  getJobPreviewConfig: (id: string) =>
    request<{ preview: Record<string, unknown>; active: boolean }>(
      `/jobs/${id}/preview-config`
    ),

  /** Replace a running job's [preview] live; optionally render one preview now. */
  updateJobPreviewConfig: (
    id: string,
    preview: Record<string, unknown>,
    previewNow = false
  ) =>
    request<{ ok: boolean; run_dir: string; preview_now: boolean }>(
      `/jobs/${id}/preview-config`,
      { method: "POST", body: JSON.stringify({ preview, preview_now: previewNow }) }
    ),

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

  /** Full library list (non-paginated); use {@link searchDatasets} for list UIs. */
  listDatasets: async () => {
    const data = await request<{ datasets?: DatasetSearchItem[] }>("/datasets");
    return data.datasets ?? [];
  },

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

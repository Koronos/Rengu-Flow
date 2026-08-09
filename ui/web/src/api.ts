import { errorMessageFromResponseBody } from "./lib/formatError";
import { filenameFromContentDisposition } from "./lib/downloadBlob";
import type { FormValues } from "./types/forms";

export interface ToolboxInput {
  param: string;
  label: string;
  control: "number" | "text" | "textarea" | "switch" | "select";
  default?: unknown;
  options?: string[];
  min?: number | null;
  max?: number | null;
  step?: number | null;
  hint?: string;
}

export interface ToolboxToolSummary {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  last_run_status: string;
}

export interface ToolboxRun {
  status: string;
  started_at?: string;
  finished_at?: string | null;
  exit_code?: number | null;
  inputs?: Record<string, unknown>;
}

export interface ToolboxTool {
  id: string;
  name: string;
  description: string;
  entrypoint: string;
  requirements: string[];
  inputs: ToolboxInput[];
  script: string;
  created_at: string;
  updated_at: string;
  last_run: ToolboxRun | null;
}

export interface ToolboxToolWrite {
  name: string;
  description?: string;
  entrypoint?: string;
  requirements?: string[];
  script?: string;
  inputs?: ToolboxInput[];
}

import type {
  CheckpointsResult,
  CompareRunsResult,
  ConfigSchemaResponse,
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
  EstimateStepsBody,
  EstimateStepsError,
  EstimateStepsResult,
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
  RunPreviewsResult,
  RunSeriesResult,
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
  QuarantineBatchInfo,
  TagBackupInfo,
  TagCommitResult,
  TagDiffResult,
  TagEditOpDto,
  TagQueryResult,
  TagSessionSummary,
  TagStatsResult,
  VersionInfo,
  PrepJobStartBody,
  PrepJobListResult,
  PrepJobReportResult,
  PrepJobRequeueBody,
  PrepModelsResult,
  PrepPromptOptions,
  PrepPromptPreviewResult,
  PrepModelDownloadResult,
  PrepStage,
  PrepCaptionConfig,
  LocalSettings,
  LocalSettingsPatch,
  QualityIndexModelsResult,
  QualityIndexStatsResult,
  QualityIndexWorstResult,
  QualityIndexCullPreviewResult,
  QualityIndexApplyResult,
} from "./types/api";
import { withDefaultPagination } from "./types/api";
import type {
  WorkflowDetail,
  WorkflowListResult,
  WorkflowNodeLogResult,
  WorkflowNodeReportResult,
  WorkflowStartOptions,
  WorkflowUpdatePayload,
  WorkflowValidateResult,
} from "./types/workflow";

const API = "/api/v1";

/** A non-2xx response, with the status code preserved so callers can distinguish an expected
 *  outcome (e.g. 409 optimistic-concurrency conflict) from a generic failure. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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
    throw new ApiError(msg || `HTTP ${res.status}`, res.status);
  }
  return data as T;
}

export const api = {
  /** renga version + git commit + installed kaon, for the sidebar version label. */
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
    request<{ preview: Record<string, unknown>; model_type?: string; active: boolean }>(
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

  /** Cross-run comparison: manifest rows + hparam columns + scalar series + timelines.
   *  `runs` is a list of run folder names; empty selects all tracked runs. */
  compareRuns: (runs: string[] = [], outputDir = "output", signal?: AbortSignal) =>
    request<CompareRunsResult>(
      `/runs/compare?runs=${encodeURIComponent(runs.join(","))}&output_dir=${encodeURIComponent(
        outputDir
      )}`,
      { signal }
    ),

  /** On-demand series for ONE metric across runs (lazy-loaded per chart in the comparison view). */
  runSeries: (
    runs: string[],
    tag: string,
    maxPoints = 500,
    outputDir = "output",
    signal?: AbortSignal
  ) =>
    request<RunSeriesResult>(
      `/runs/series?runs=${encodeURIComponent(runs.join(","))}&tag=${encodeURIComponent(
        tag
      )}&max_points=${maxPoints}&output_dir=${encodeURIComponent(outputDir)}`,
      { signal }
    ),

  /** Lazy preview-frame list for one run (compare view media), abortable on unmount. */
  runPreviews: (name: string, outputDir = "output", signal?: AbortSignal) =>
    request<RunPreviewsResult>(
      `/runs/${encodeURIComponent(name)}/previews?output_dir=${encodeURIComponent(outputDir)}`,
      { signal }
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

  /** Read rengu.local.toml settings (editable + restart-required + read-only groups). */
  getSettings: () => request<LocalSettings>("/settings"),

  /** Write the editable subset of rengu.local.toml; returns the re-read settings. */
  updateSettings: (patch: LocalSettingsPatch) =>
    request<LocalSettings>("/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),

  // --- Dataset prep: tag editor ---

  prepOpenTagSession: (path: string, format: string, ext: string) =>
    request<TagSessionSummary>("/prep/tags/sessions", {
      method: "POST",
      body: JSON.stringify({ path, format, ext }),
    }),

  prepTagSessionSummary: (sessionId: string) =>
    request<TagSessionSummary>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}`),

  prepTagStats: (sessionId: string, scope: string) =>
    request<TagStatsResult>(
      `/prep/tags/sessions/${encodeURIComponent(sessionId)}/stats?scope=${encodeURIComponent(scope)}`
    ),

  prepTagQuery: (
    sessionId: string,
    filter: TagEditOpDto["filter"],
    scope: string,
    limit?: number,
    offset?: number
  ) =>
    request<TagQueryResult>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}/query`, {
      method: "POST",
      body: JSON.stringify({
        filter,
        scope,
        ...(limit != null ? { limit } : {}),
        ...(offset != null ? { offset } : {}),
      }),
    }),

  prepTagSizeQuery: (
    sessionId: string,
    body: { below?: number; above?: number },
    limit?: number,
    offset?: number
  ) =>
    request<TagQueryResult>(
      `/prep/tags/sessions/${encodeURIComponent(sessionId)}/size-query`,
      {
        method: "POST",
        body: JSON.stringify({
          ...body,
          ...(limit != null ? { limit } : {}),
          ...(offset != null ? { offset } : {}),
        }),
      }
    ),

  prepStageTagOps: (sessionId: string, ops: TagEditOpDto[]) =>
    request<TagSessionSummary>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}/ops`, {
      method: "POST",
      body: JSON.stringify({ ops }),
    }),

  prepUndoTagOp: (sessionId: string) =>
    request<TagSessionSummary>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}/undo`, {
      method: "POST",
    }),

  prepTagDiff: (sessionId: string, limit?: number) =>
    request<TagDiffResult>(
      `/prep/tags/sessions/${encodeURIComponent(sessionId)}/diff${limit ? `?limit=${limit}` : ""}`
    ),

  prepCommitTagSession: (sessionId: string) =>
    request<TagCommitResult>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}/commit`, {
      method: "POST",
    }),

  prepCloseTagSession: (sessionId: string) =>
    request<{ ok: boolean }>(`/prep/tags/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),

  prepTagBackups: (path: string) =>
    request<{ backups: TagBackupInfo[] }>(
      `/prep/tags/backups?path=${encodeURIComponent(path)}`
    ),

  prepRestoreTagBackup: (path: string, backup: string) =>
    request<{ restored: string[] }>("/prep/tags/restore", {
      method: "POST",
      body: JSON.stringify({ path, backup }),
    }),

  prepQuarantineBatches: (path: string) =>
    request<{ batches: QuarantineBatchInfo[] }>(
      `/prep/tags/quarantine?path=${encodeURIComponent(path)}`
    ),

  prepRestoreQuarantine: (path: string, batch: string) =>
    request<{ restored: string[] }>("/prep/tags/quarantine/restore", {
      method: "POST",
      body: JSON.stringify({ path, batch }),
    }),

  // --- Dataset prep: jobs ---

  /** List prep jobs (kind=prep filter). Existing listJobs() remains unchanged (train-only server-side default). */
  prepJobs: () => request<PrepJobListResult>("/jobs?kind=prep"),

  /** Create a new prep job (tag | caption | clean). */
  createPrepJob: (body: PrepJobStartBody) =>
    request<JobRecord>("/prep/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Fetch stage report for a finished/running prep job. */
  prepJobReport: (id: string) =>
    request<PrepJobReportResult>(`/prep/jobs/${encodeURIComponent(id)}/report`),

  /** Parsed config of a prep job, to seed a new job from it. */
  prepJobConfig: (id: string) =>
    request<{ config: Record<string, unknown> }>(
      `/prep/jobs/${encodeURIComponent(id)}/config`
    ),

  /** List available models for a prep stage. */
  prepModels: (stage: PrepStage) =>
    request<PrepModelsResult>(`/prep/models?stage=${encodeURIComponent(stage)}`),

  /** Trigger download of a prep model. */
  prepCaptionPrompts: () =>
    request<PrepPromptOptions>("/prep/caption-prompts"),

  /** Render the exact prompt text the job will send to the model (server-side composition). */
  prepCaptionPromptPreview: (caption: Partial<PrepCaptionConfig>, sampleTags?: string[]) =>
    request<PrepPromptPreviewResult>("/prep/caption-prompts/preview", {
      method: "POST",
      body: JSON.stringify({ caption, sample_tags: sampleTags }),
    }),

  prepModelDownload: (stage: PrepStage, modelId: string) =>
    request<PrepModelDownloadResult>("/prep/models/download", {
      method: "POST",
      body: JSON.stringify({ stage, model_id: modelId }),
    }),

  /** Re-queue a stopped/failed/finished prep job. */
  requeuePrepJob: (id: string, body: PrepJobRequeueBody) =>
    request<JobRecord>(`/prep/jobs/${encodeURIComponent(id)}/requeue`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // --- Toolbox ---

  toolboxEnabled: () => request<{ enabled: boolean }>("/toolbox/enabled"),
  listToolboxTools: () => request<ToolboxToolSummary[]>("/toolbox/tools"),
  createToolboxTool: (body: ToolboxToolWrite) =>
    request<ToolboxTool>("/toolbox/tools", { method: "POST", body: JSON.stringify(body) }),
  getToolboxTool: (id: string) => request<ToolboxTool>(`/toolbox/tools/${id}`),
  updateToolboxTool: (id: string, body: Partial<ToolboxToolWrite>) =>
    request<ToolboxTool>(`/toolbox/tools/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteToolboxTool: (id: string) =>
    request<{ ok: boolean }>(`/toolbox/tools/${id}`, { method: "DELETE" }),
  runToolboxTool: (id: string, values: Record<string, unknown>) =>
    request<ToolboxRun>(`/toolbox/tools/${id}/run`, {
      method: "POST",
      body: JSON.stringify({ values }),
    }),
  toolboxRunStatus: (id: string) => request<ToolboxRun>(`/toolbox/tools/${id}/run`),
  toolboxLog: (id: string, offset = 0) =>
    request<{ chunk: string; offset: number; status: string }>(
      `/toolbox/tools/${id}/log?offset=${offset}`,
    ),
  cancelToolboxRun: (id: string) =>
    request<{ ok: boolean }>(`/toolbox/tools/${id}/run/cancel`, { method: "POST" }),

  /** Estimate total training steps for a run config + dataset (no server-side disk scan by default). */
  estimateSteps: (body: EstimateStepsBody) =>
    request<EstimateStepsResult | EstimateStepsError>("/runs/estimate-steps", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // --- Dataset prep: quality index ---

  /** List every quality model that has scores recorded for a folder. */
  qualityIndexModels: (path: string) =>
    request<QualityIndexModelsResult>(
      `/prep/quality-index/models?path=${encodeURIComponent(path)}`
    ),

  /** Live stats for one quality model in a dataset (reflects current state after culls). */
  qualityIndexStats: (path: string, model: string) =>
    request<QualityIndexStatsResult>(
      `/prep/quality-index/stats?path=${encodeURIComponent(path)}&model=${encodeURIComponent(model)}`
    ),

  /** Worst-scoring images for one quality model (lowest scores first). */
  qualityIndexWorst: (params: { path: string; model: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams({ path: params.path, model: params.model });
    if (params.limit != null) q.set("limit", String(params.limit));
    if (params.offset != null) q.set("offset", String(params.offset));
    return request<QualityIndexWorstResult>(`/prep/quality-index/worst?${q.toString()}`);
  },

  /** Preview how many images would be culled for each model at the given percentile thresholds. */
  qualityIndexCullPreview: (body: { path: string; per_model: Record<string, number> }) =>
    request<QualityIndexCullPreviewResult>("/prep/quality-index/cull-preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Move images flagged by the given per-model percentile thresholds to <path>/low_quality/. */
  qualityIndexApply: (body: {
    path: string;
    per_model: Record<string, number>;
    output_dir?: string;
  }) =>
    request<QualityIndexApplyResult>("/prep/quality-index/apply", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // --- Workflows ---

  listWorkflows: () => request<WorkflowListResult>("/workflows"),

  /** `PUT /workflows/{id}` is the only place a graph can be written — create only names the row. */
  createWorkflow: (name = "") =>
    request<WorkflowDetail>("/workflows", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  getWorkflow: (id: number | string) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(String(id))}`),

  /**
   * Save the graph under optimistic concurrency: `version` must match the server's current one.
   *
   * Throws `ApiError` with `status === 409` when it doesn't (another tab/save landed first) or
   * while the workflow is `running`/`cancelling` — an EXPECTED outcome, not a generic failure.
   * Catch it and offer to reload the workflow rather than showing a plain error toast.
   */
  updateWorkflow: (id: number | string, payload: WorkflowUpdatePayload) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(String(id))}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  /** Also 409s (`ApiError`) while the workflow is `running`/`cancelling` — stop it first. */
  deleteWorkflow: (id: number | string) =>
    request<{ ok: boolean }>(`/workflows/${encodeURIComponent(String(id))}`, {
      method: "DELETE",
    }),

  cloneWorkflow: (id: number | string, name?: string) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(String(id))}/clone`, {
      method: "POST",
      body: JSON.stringify(name != null ? { name } : {}),
    }),

  validateWorkflow: (id: number | string) =>
    request<WorkflowValidateResult>(
      `/workflows/${encodeURIComponent(String(id))}/validate`,
      { method: "POST" }
    ),

  startWorkflow: (id: number | string, options?: WorkflowStartOptions) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(String(id))}/start`, {
      method: "POST",
      body: JSON.stringify({
        from_node: options?.from_node ?? null,
        force: options?.force ?? false,
        only: options?.only ?? false,
      }),
    }),

  cancelWorkflow: (id: number | string) =>
    request<WorkflowDetail>(`/workflows/${encodeURIComponent(String(id))}/cancel`, {
      method: "POST",
    }),

  workflowNodeLog: (workflowId: number | string, nodeId: string, offset = 0) =>
    request<WorkflowNodeLogResult>(
      `/workflows/${encodeURIComponent(String(workflowId))}/nodes/${encodeURIComponent(
        nodeId
      )}/log?offset=${offset}`
    ),

  /** The node's `report.json` (prep stages) or `result.json` (tools) — the Output tab's data. */
  workflowNodeReport: (workflowId: number | string, nodeId: string) =>
    request<WorkflowNodeReportResult>(
      `/workflows/${encodeURIComponent(String(workflowId))}/nodes/${encodeURIComponent(
        nodeId
      )}/report`
    ),
};

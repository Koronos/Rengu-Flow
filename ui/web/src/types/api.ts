/** Typed shapes for `/api/v1` JSON (control-plane server). */

import type { FormValues } from "./forms";

export type QueryParams =
  | URLSearchParams
  | Record<string, string | number | boolean | null | undefined>;

export function toSearchParams(params: QueryParams): URLSearchParams {
  if (params instanceof URLSearchParams) return params;
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") q.set(key, String(value));
  }
  return q;
}

export function withDefaultPagination(
  params: QueryParams,
  defaults: { page?: string; page_size?: string } = {}
): URLSearchParams {
  const q = toSearchParams(params);
  if (defaults.page && !q.has("page")) q.set("page", defaults.page);
  if (defaults.page_size && !q.has("page_size")) q.set("page_size", defaults.page_size);
  return q;
}

// --- Shared ---

export interface ApiOk {
  ok?: boolean;
}

export interface ValidateResult {
  ok: boolean;
  error?: string;
  warnings?: string[];
}

export interface ValidateOnlyResult {
  ok: boolean;
  returncode?: number;
  stdout?: string;
  stderr?: string;
  error?: string;
}

export interface ParseTomlResult {
  form?: FormValues;
  error?: string;
}

export interface RenderTomlResult {
  content?: string;
  error?: string;
}

export interface Paginated<T> {
  items: T[];
  total: number;
  page?: number;
  page_size?: number;
}

// --- Library (datasets) shared shapes ---

export interface DuplicateConfigResult {
  id: number | string;
}

export interface ImportExampleResult {
  id: number | string;
}

export interface ImportConfigResult {
  id: number | string;
}

export interface ExportBundleResult {
  blob: Blob;
  filename: string;
}

// --- Datasets ---

export interface DatasetMeta {
  id: number | string;
  name?: string;
  folder_count?: number;
  created_at?: string;
  updated_at?: string;
}

export interface DatasetDetail {
  id: number | string;
  content?: string;
  meta?: DatasetMeta;
}

export interface DatasetSearchItem extends DatasetMeta {
  preview?: Record<string, unknown>;
  path?: string;
  directory_count?: number;
}

export interface DatasetSavePayload {
  content?: string;
  form?: FormValues;
  id?: string;
}

export interface DatasetComposeResult {
  id: number | string;
  dataset_ref?: string;
  preview?: Record<string, unknown>;
}

export interface DatasetPreviewResult {
  directories?: { path?: string; image_count?: number }[];
  error?: string;
}

export interface DatasetPreviewImage {
  token: string;
  name?: string;
  directory_index?: number;
}

export interface DatasetPreviewImagesResult {
  images?: DatasetPreviewImage[];
  total?: number;
  has_more?: boolean;
}

export interface DatasetScanPathResult {
  ok?: boolean;
  path?: string;
  error?: string;
  image_count?: number;
  video_count?: number;
  caption_txt_files?: number;
  has_captions_json?: boolean;
  count_capped?: boolean;
  image_count_display?: string;
  total_media?: number;
}

export interface FsStatResult {
  exists: boolean;
  is_file: boolean;
  is_dir: boolean;
  resolved_path?: string;
  error?: string;
}

export interface DatasetFolderSuggestion {
  path: string;
  label?: string;
}

// --- Jobs & training hub ---

export type JobState =
  | "new"
  | "pending"
  | "running"
  | "stopping"
  | "finished"
  | "failed"
  | "stopped";

export interface JobRecord {
  id: string;
  run_name?: string | null;
  config_path?: string;
  state: JobState | string;
  pid?: number | null;
  run_dir?: string | null;
  output_dir?: string;
  num_gpus?: number;
  resume_from?: string | null;
  log_path?: string;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  extra_args?: string[];
  queue_position?: number | null;
  source_run_dir?: string | null;
  cache_only?: boolean;
  trust_cache?: boolean;
  regenerate_cache?: boolean;
  status?: RunStatusFile | null;
  signals_available?: boolean;
  /** Immutable snapshot of the run's own config TOML (library refs intact). */
  config_content?: string;
}

export interface JobListResult {
  jobs: JobRecord[];
  stats: { running: number; pending: number };
}

export interface JobStartBody {
  /** Inline config TOML for the run (the run carries its own snapshot). */
  content?: string;
  num_gpus?: number;
  resume_from?: string;
  output_dir?: string;
  extra_args?: string;
  reset_dataloader?: boolean;
  reset_optimizer?: boolean;
  /** Run dataset cache step only (--cache_only); no training. */
  cache_only?: boolean;
  /** Skip cache rebuild when cache already exists (--trust_cache). */
  trust_cache?: boolean;
  /** Force full cache rebuild (--regenerate_cache). */
  regenerate_cache?: boolean;
  enqueue?: boolean;
  start_immediately?: boolean;
  /** Save as a `new` draft instead of queuing. */
  save_for_later?: boolean;
  source_run_dir?: string;
}

export interface CloneJobBody {
  num_gpus?: number;
  output_dir?: string;
  extra_args?: string;
  start_immediately?: boolean;
}

export interface JobPatchBody {
  state?: string;
  [key: string]: unknown;
}

export interface JobImportBody {
  run_path: string;
  dataset_id?: number | string;
  import_dataset?: boolean;
  allow_duplicate?: boolean;
}

export interface ContinueRunBody {
  run_path: string;
  content: string;
  num_gpus?: number;
  extra_args?: string;
  reset_dataloader?: boolean;
  reset_optimizer?: boolean;
  /** Checkpoint folder name to resume from; omit to use the run's `latest`. */
  resume_from?: string;
  /** Ignore checkpoints and train from step 0 in the same folder. */
  from_scratch?: boolean;
  enqueue?: boolean;
  start_immediately?: boolean;
}

export interface CheckpointInfo {
  name: string;
  step: number;
  is_latest: boolean;
  /** Saved after `latest` — may be truncated/corrupt (e.g. disk filled). */
  suspect: boolean;
}

export interface CheckpointsResult {
  checkpoints: CheckpointInfo[];
  run_dir: string | null;
}

export interface RunProgress {
  step?: number | null;
  max_steps?: number | null;
  epoch?: number | null;
  epochs?: number | null;
  loss?: number | null;
  percent?: number | null;
  steps_remaining?: number | null;
  steps_per_second?: number | null;
  steps_per_second_ema?: number | null;
  eta_sec?: number | null;
  /** Human-readable ETA from trainer, e.g. `1h 23m` */
  eta?: string | null;
  /** e.g. `training`, `waiting_disk_export` (paused for disk during model export) */
  phase?: string | null;
  updated_at?: string | null;
  status_available?: boolean;
  model_type?: string;
  run_name_label?: string;
}

export type TrainingRunKind = "job";

export interface TrainingRunRow {
  key: string;
  kind: TrainingRunKind;
  job_id?: string | null;
  state: string;
  run_dir?: string | null;
  run_name?: string | null;
  label?: string | null;
  output_dir?: string | null;
  num_gpus?: number | null;
  resume_from?: string | null;
  queue_position?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  cache_only?: boolean;
  trust_cache?: boolean;
  regenerate_cache?: boolean;
  progress?: RunProgress | null;
  has_tensorboard?: boolean;
  scalars?: Record<string, { step: number; value: number }[]>;
  preview_images?: RunPreviewImageRef[];
}

export interface TrainRunsResult {
  items: TrainingRunRow[];
  total: number;
  page?: number;
  page_size?: number;
  stats: { running: number; pending: number };
}

export interface TrainActiveResult {
  active: TrainingRunRow | null;
}

export interface ImportRunPreview {
  already_imported?: boolean;
  config_path?: string;
  run?: FsRunRecord;
  suggested_config_id?: string | number;
  suggested_dataset_id?: string | number;
}

export interface ImportCandidatesResult {
  runs?: { path: string; name?: string; already_imported?: boolean }[];
}

// --- Filesystem runs ---

export interface RunStatusFile {
  step?: number;
  loss?: number;
  epoch?: number;
  phase?: string;
  updated_at?: string;
}

export interface FsRunRecord {
  path?: string;
  name?: string;
  status?: RunStatusFile;
  artifacts?: Record<string, unknown>[];
  has_tensorboard?: boolean;
  signals_available?: boolean;
}

export interface TrainingSignalDef {
  id: string;
  label: string;
  group: string;
  hint?: string;
  disk_wait_only?: boolean;
  variant?: "primary" | "danger";
}

export interface TrainingSignalsResult {
  signals: TrainingSignalDef[];
  active_job_states: string[];
}

export interface FsRunsListResult {
  runs?: FsRunRecord[];
}

export interface RunConfigResult {
  content?: string;
  path?: string;
}

// --- System stats ---

export interface SystemStatsSummaryGpu {
  index: number;
  util_percent?: number | null;
  vram_used_gb?: number | null;
  vram_total_gb?: number | null;
  vram_percent?: number | null;
  temp_c?: number | null;
}

export interface SystemStatsSummary {
  cpu_percent?: number | null;
  cpu_temp_c?: number | null;
  ram_percent?: number | null;
  ram_used_gb?: number | null;
  ram_total_gb?: number | null;
  gpus?: SystemStatsSummaryGpu[];
}

export interface CpuFreqMhz {
  current?: number | null;
  min?: number | null;
  max?: number | null;
}

export interface CpuDetail {
  available?: boolean;
  percent?: number | null;
  per_core?: number[];
  logical_count?: number;
  physical_count?: number;
  freq_mhz?: CpuFreqMhz;
  temperatures?: TemperatureReading[];
  temp_c?: number | null;
  error?: string;
}

export interface RamSwapDetail {
  percent?: number | null;
  used_gb?: number | null;
  total_gb?: number | null;
}

export interface RamDetail {
  available?: boolean;
  percent?: number | null;
  used_gb?: number | null;
  total_gb?: number | null;
  swap?: RamSwapDetail;
  error?: string;
}

export interface TemperatureReading {
  label?: string;
  current_c?: number | null;
  high_c?: number | null;
}

export interface GpuDeviceDetail {
  index?: number;
  name?: string;
  util_percent?: number | null;
  vram_percent?: number | null;
  vram_used_gb?: number | null;
  vram_total_gb?: number | null;
  temp_c?: number | null;
  power_w?: number | null;
  fan_percent?: number | null;
  clock_sm_mhz?: number | null;
}

export interface GpusDetail {
  error?: string;
  devices?: GpuDeviceDetail[];
}

export interface SystemStatsDetail {
  cpu?: CpuDetail;
  ram?: RamDetail;
  gpus?: GpusDetail;
  temperatures?: TemperatureReading[];
  warnings?: string[];
}

export interface SystemStatsResponse {
  ok?: boolean;
  ts?: number;
  summary?: SystemStatsSummary;
  detail?: SystemStatsDetail;
}

// --- Registry probe ---

export interface RegistryProbeTarget {
  available?: boolean;
  resolved_class?: string;
  resolved?: string;
  source?: string;
  error?: string;
}

export interface RegistryProbeResult {
  optimizer?: RegistryProbeTarget;
  scheduler?: RegistryProbeTarget;
}

export interface RegistryProbeBody {
  optimizer?: string;
  scheduler?: string;
}

// --- Docs ---

export interface DocIndexItem {
  path: string;
  title?: string;
}

export interface DocsIndexResult {
  items: DocIndexItem[];
}

export interface DocContentResult {
  path: string;
  content: string;
}

// --- TensorBoard ---

export interface TensorboardStatus {
  running?: boolean;
  url?: string;
  error?: string;
}

export interface TensorboardStartBody {
  output_dir?: string;
  port?: number;
  host?: string;
}

export interface TensorboardStartResult {
  running?: boolean;
  url?: string;
}

// --- Schema (open sections; field defs in forms.ts) ---

export interface ConfigSchemaResponse {
  sections?: unknown[];
  registries?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DatasetSchemaResponse {
  sections?: unknown[];
  registries?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AugmentationCatalogResponse {
  version?: string;
  seed_modes?: string[];
  variant_sampling_modes?: string[];
  presets?: {
    name: string;
    label: string;
    available: boolean;
    deferred?: boolean;
    strategies: string[];
    strategy_defaults?: Record<string, Record<string, unknown>>;
  }[];
  strategies?: {
    name: string;
    label: string;
    category: string;
    implemented: boolean;
    enumerable: boolean;
    parameters: {
      path: string;
      label: string;
      type: string;
      default?: number | string | boolean;
      min?: number;
      max?: number;
      step?: number;
      options?: string[];
    }[];
  }[];
}

// --- Metrics ---

export interface ScalarMetricPoint {
  step: number;
  value: number;
  wall_time?: number;
}

export interface RunPreviewImageRef {
  run_dir: string;
  name: string;
}

export interface JobMetricsResult {
  scalars?: Record<string, ScalarMetricPoint[]>;
  preview_images?: RunPreviewImageRef[];
  previews?: { name: string; path?: string }[];
}

// --- Maintenance ---

export interface MaintenanceEnabledResult {
  enabled: boolean;
}

export interface MaintenanceDbInfo {
  path: string;
  exists: boolean;
  size_bytes: number;
  modified_at: number | null;
  tables: string[];
  expected_tables: string[];
}

export interface MaintenanceDepProfile {
  id: string;
  label: string;
  description: string;
  command: string;
  pip_allowed: boolean;
}

export interface MaintenanceStatus {
  enabled: boolean;
  schema_version: number;
  database: MaintenanceDbInfo;
  ui_data_dir: string;
  repo_root: string;
  python_executable: string;
  git: {
    available: boolean;
    gitmodules_exists: boolean;
    submodules_configured: { path?: string; url?: string }[];
    submodule_status: { path: string; status: string; commit: string; line: string }[];
  };
  requirements_files: { path: string; exists: boolean; resolved: string | null }[];
  pyproject_exists: boolean;
  dependency_profiles: MaintenanceDepProfile[];
  pip_install_from_server: boolean;
}

export interface MaintenanceCommandOutput {
  ok: boolean;
  message?: string;
  stdout?: string;
  stderr?: string;
  returncode?: number;
  command?: string | string[];
  executed?: boolean;
  profile?: string;
}

export interface MaintenanceDbResetResult extends MaintenanceCommandOutput {
  path: string;
  tables_before: string[];
  tables_after: string[];
}

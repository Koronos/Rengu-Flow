/** Loose API / UI row shapes (server JSON). */

export type JsonRecord = Record<string, unknown>;

export interface JobRecord extends JsonRecord {
  id?: string;
  run_dir?: string;
  output_dir?: string;
  state?: string;
  pid?: number;
}

export interface RunStatusFile extends JsonRecord {
  step?: number;
  loss?: number;
}

export interface FsRunRecord extends JsonRecord {
  path?: string;
  name?: string;
  status?: RunStatusFile;
  artifacts?: JsonRecord[];
  has_tensorboard?: boolean;
}

export interface DocIndexItem {
  path: string;
  title?: string;
}

export interface ImportRunPreview extends JsonRecord {
  already_imported?: boolean;
  config_path?: string;
  run?: FsRunRecord;
  suggested_config_id?: string;
  suggested_dataset_id?: string;
}

export interface TensorboardStatus extends JsonRecord {
  running?: boolean;
  url?: string;
}

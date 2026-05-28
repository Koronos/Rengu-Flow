/** Re-exports for views/stores — prefer `./api` for new code. */

/** Untyped JSON object from legacy call sites; prefer concrete API types. */
export type JsonRecord = Record<string, unknown>;

export type {
  DocIndexItem,
  FsRunRecord,
  ImportRunPreview,
  JobRecord,
  RunStatusFile,
  TensorboardStatus,
  TrainingRunRow,
} from "./api";

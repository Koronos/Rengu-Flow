import { describe, expect, it } from "vitest";
import {
  ACTIVE_JOB_STATES,
  fsRunSignalsAvailable,
  groupSignalDefinitions,
  jobSignalsAvailable,
} from "./trainingSignals";
import type { TrainingSignalDef } from "../types/api";

const SAMPLE_DEFS: TrainingSignalDef[] = [
  { id: "save", label: "Checkpoint", group: "Resume checkpoint" },
  { id: "continue", label: "Continue export", group: "Disk recovery", disk_wait_only: true },
  { id: "quit", label: "Quit without save", group: "Disk recovery", disk_wait_only: true },
  { id: "preview", label: "Preview", group: "Preview" },
];

describe("groupSignalDefinitions", () => {
  it("hides disk-wait-only signals unless export is paused", () => {
    const normal = groupSignalDefinitions(SAMPLE_DEFS, false);
    expect(normal.flatMap((g) => g.items.map((i) => i.id))).toEqual(["save", "preview"]);

    const waiting = groupSignalDefinitions(SAMPLE_DEFS, true);
    expect(waiting.flatMap((g) => g.items.map((i) => i.id))).toEqual([
      "save",
      "continue",
      "quit",
      "preview",
    ]);
  });
});

describe("jobSignalsAvailable", () => {
  it("prefers backend signals_available flag", () => {
    expect(jobSignalsAvailable({ state: "running", signals_available: false })).toBe(false);
    expect(jobSignalsAvailable({ state: "stopped", signals_available: true })).toBe(true);
  });

  it("falls back to active job states", () => {
    for (const state of ACTIVE_JOB_STATES) {
      expect(jobSignalsAvailable({ state })).toBe(true);
    }
    expect(jobSignalsAvailable({ state: "stopped" })).toBe(false);
    expect(jobSignalsAvailable({ state: "finished" })).toBe(false);
  });
});

describe("fsRunSignalsAvailable", () => {
  it("uses backend signals_available only", () => {
    expect(fsRunSignalsAvailable({ signals_available: true })).toBe(true);
    expect(fsRunSignalsAvailable({ signals_available: false })).toBe(false);
    expect(fsRunSignalsAvailable({})).toBe(false);
  });
});

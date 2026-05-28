import { describe, expect, it } from "vitest";
import {
  directoryFieldWritesToToml,
  globalFieldDisplayHint,
  initialValueForOptionalField,
  setOverrideEnabled,
} from "./datasetDirectoryForm";
import type { SchemaField } from "../types/forms";

const subsampleField: SchemaField = {
  path: "subsample_ratio",
  label: "Subsample ratio",
  type: "number",
  default: 1,
  show_if_set: true,
};

describe("subsample_ratio defaults", () => {
  it("initializes to schema default 1 when enabled", () => {
    expect(initialValueForOptionalField(subsampleField)).toBe(1);
    expect(initialValueForOptionalField(subsampleField)).not.toBe(0.25);
  });

  it("setOverrideEnabled uses the same starter value", () => {
    const row = setOverrideEnabled(
      subsampleField,
      { path: "/data", num_repeats: 1 },
      true
    );
    expect(row.subsample_ratio).toBe(1);
    expect(row.num_repeats).toBe(1);
  });

  it("does not copy num_repeats when enabling subsample_ratio", () => {
    const row = setOverrideEnabled(
      subsampleField,
      { path: "/data", num_repeats: 3 },
      true
    );
    expect(row.subsample_ratio).toBe(1);
    expect(row.num_repeats).toBe(3);
  });

  it("global hint shows 1 when dataset default is full dataset", () => {
    expect(globalFieldDisplayHint(subsampleField, { subsample_ratio: 1 })).toBe("1");
  });

  it("does not write directory subsample_ratio when unset (inherits global 1)", () => {
    expect(
      directoryFieldWritesToToml(subsampleField, { path: "/data", num_repeats: 1 })
    ).toBe(false);
  });
});

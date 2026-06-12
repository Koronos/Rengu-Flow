import { describe, expect, it } from "vitest";
import {
  directoryFieldWritesToToml,
  directoryOverrideBlockVisible,
  globalFieldDisplayHint,
  initialValueForOptionalField,
  isOverrideEnabled,
  needsDirectoryOverrideToggle,
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

const maxImagesField: SchemaField = {
  path: "max_images",
  label: "Max images per epoch",
  type: "integer",
  min: 1,
  show_if_set: true,
};

const subsampleShuffleField: SchemaField = {
  path: "subsample_shuffle",
  label: "Rotate subsampled window",
  type: "boolean",
  default: true,
  show_if_set: true,
};

describe("max_images directory override", () => {
  it("initializes the integer override to its min (1) when enabled", () => {
    expect(initialValueForOptionalField(maxImagesField)).toBe(1);
  });

  it("setOverrideEnabled adds max_images without touching num_repeats", () => {
    const row = setOverrideEnabled(maxImagesField, { path: "/data", num_repeats: 2 }, true);
    expect(row.max_images).toBe(1);
    expect(row.num_repeats).toBe(2);
  });

  it("setOverrideEnabled(false) removes max_images so the row inherits again", () => {
    const row = setOverrideEnabled(
      maxImagesField,
      { path: "/data", num_repeats: 1, max_images: 10 },
      false
    );
    expect("max_images" in row).toBe(false);
  });

  it("does not write max_images when unset", () => {
    expect(
      directoryFieldWritesToToml(maxImagesField, { path: "/data", num_repeats: 1 })
    ).toBe(false);
  });

  it("subsample_shuffle defaults to true and is an optional override", () => {
    expect(initialValueForOptionalField(subsampleShuffleField)).toBe(true);
    expect(needsDirectoryOverrideToggle(subsampleShuffleField)).toBe(true);
    const row = setOverrideEnabled(subsampleShuffleField, { path: "/data", num_repeats: 1 }, true);
    expect("subsample_shuffle" in row).toBe(true);
    expect(
      directoryFieldWritesToToml(subsampleShuffleField, { path: "/data", num_repeats: 1 })
    ).toBe(false);
  });
});

const onlineCaptionsField: SchemaField = {
  path: "online_captions",
  label: "Online captions.json",
  type: "boolean",
  default: false,
};

const minArField: SchemaField = {
  path: "min_ar",
  label: "Min aspect ratio",
  type: "number",
  default: 0.5,
  show_when_field: "enable_ar_bucket",
};

describe("directory override toggles", () => {
  it("needs toggle for explicit per-directory fields like online_captions", () => {
    expect(needsDirectoryOverrideToggle(onlineCaptionsField)).toBe(true);
    expect(needsDirectoryOverrideToggle({ path: "path", label: "Path", type: "string" })).toBe(
      false
    );
  });

  it("isOverrideEnabled is false until the key exists on the row", () => {
    expect(isOverrideEnabled(onlineCaptionsField, { path: "/data", num_repeats: 1 })).toBe(false);
    expect(
      isOverrideEnabled(
        onlineCaptionsField,
        setOverrideEnabled(onlineCaptionsField, { path: "/data", num_repeats: 1 }, true)
      )
    ).toBe(true);
  });

  it("setOverrideEnabled(false) removes the key so the row inherits again", () => {
    const row = setOverrideEnabled(
      onlineCaptionsField,
      { path: "/data", num_repeats: 1, online_captions: true },
      false
    );
    expect("online_captions" in row).toBe(false);
    expect(isOverrideEnabled(onlineCaptionsField, row)).toBe(false);
  });

  it("shows min_ar when global enable_ar_bucket is on", () => {
    expect(
      directoryOverrideBlockVisible(
        minArField,
        { path: "/data", num_repeats: 1 },
        { enable_ar_bucket: true }
      )
    ).toBe(true);
    expect(
      directoryOverrideBlockVisible(
        minArField,
        { path: "/data", num_repeats: 1 },
        { enable_ar_bucket: false }
      )
    ).toBe(false);
  });
});

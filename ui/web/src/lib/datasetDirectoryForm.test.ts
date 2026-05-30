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

const staticSamplingField: SchemaField = {
  path: "static_sampling",
  label: "Static sampling (no rotation)",
  type: "boolean",
  default: false,
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

  it("static_sampling defaults to false and is an optional override", () => {
    expect(initialValueForOptionalField(staticSamplingField)).toBe(false);
    expect(needsDirectoryOverrideToggle(staticSamplingField)).toBe(true);
    const row = setOverrideEnabled(staticSamplingField, { path: "/data", num_repeats: 1 }, true);
    expect("static_sampling" in row).toBe(true);
    expect(
      directoryFieldWritesToToml(staticSamplingField, { path: "/data", num_repeats: 1 })
    ).toBe(false);
  });
});

const shuffleTagsField: SchemaField = {
  path: "shuffle_tags",
  label: "Shuffle tags",
  type: "boolean",
  default: false,
};

const cacheShuffleField: SchemaField = {
  path: "cache_shuffle_num",
  label: "Cache shuffle count",
  type: "integer",
  default: 1,
  show_when_field: "shuffle_tags",
};

describe("directory override toggles", () => {
  it("needs toggle for explicit per-directory fields like shuffle_tags", () => {
    expect(needsDirectoryOverrideToggle(shuffleTagsField)).toBe(true);
    expect(needsDirectoryOverrideToggle({ path: "path", label: "Path", type: "string" })).toBe(
      false
    );
  });

  it("isOverrideEnabled is false until the key exists on the row", () => {
    expect(isOverrideEnabled(shuffleTagsField, { path: "/data", num_repeats: 1 })).toBe(false);
    expect(
      isOverrideEnabled(
        shuffleTagsField,
        setOverrideEnabled(shuffleTagsField, { path: "/data", num_repeats: 1 }, true)
      )
    ).toBe(true);
  });

  it("setOverrideEnabled(false) removes the key so the row inherits again", () => {
    const row = setOverrideEnabled(
      shuffleTagsField,
      { path: "/data", num_repeats: 1, shuffle_tags: true },
      false
    );
    expect("shuffle_tags" in row).toBe(false);
    expect(isOverrideEnabled(shuffleTagsField, row)).toBe(false);
  });

  it("shows cache_shuffle_num when global shuffle_tags is on", () => {
    expect(
      directoryOverrideBlockVisible(
        cacheShuffleField,
        { path: "/data", num_repeats: 1 },
        { shuffle_tags: true }
      )
    ).toBe(true);
    expect(
      directoryOverrideBlockVisible(
        cacheShuffleField,
        { path: "/data", num_repeats: 1 },
        { shuffle_tags: false }
      )
    ).toBe(false);
  });
});

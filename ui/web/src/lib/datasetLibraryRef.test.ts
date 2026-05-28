import { describe, expect, it } from "vitest";
import {
  appendUniqueDatasetPaths,
  canonicalDatasetRef,
  datasetRefDisplayLabel,
  formatDatasetLibraryRef,
  isLibraryDatasetRef,
  libraryDatasetIdFromRef,
  parseDatasetLibraryRef,
} from "./datasetLibraryRef";

describe("datasetLibraryRef", () => {
  it("parses ref with display suffix", () => {
    const ref = "renga-flow-dataset:3:artista 1";
    expect(isLibraryDatasetRef(ref)).toBe(true);
    const p = parseDatasetLibraryRef(ref);
    expect(p.id).toBe("3");
    expect(p.label).toBe("artista 1");
    expect(canonicalDatasetRef(ref)).toBe("renga-flow-dataset:3");
  });

  it("parses ref without suffix", () => {
    const ref = "renga-flow-dataset:12";
    const p = parseDatasetLibraryRef(ref);
    expect(p.id).toBe("12");
    expect(p.label).toBeNull();
  });

  it("formats ref with optional label", () => {
    expect(formatDatasetLibraryRef(5, "My set")).toBe("renga-flow-dataset:5:My set");
    expect(formatDatasetLibraryRef(5)).toBe("renga-flow-dataset:5");
  });

  it("display label prefers suffix then id", () => {
    expect(datasetRefDisplayLabel("renga-flow-dataset:3:artista 1")).toBe("artista 1");
    expect(datasetRefDisplayLabel("renga-flow-dataset:12")).toBe("12");
    expect(datasetRefDisplayLabel("/data/foo.toml")).toBe("/data/foo.toml");
  });

  it("libraryDatasetIdFromRef validates numeric id", () => {
    expect(libraryDatasetIdFromRef("renga-flow-dataset:3:label")).toBe("3");
    expect(libraryDatasetIdFromRef("renga-flow-dataset:abc")).toBeNull();
    expect(libraryDatasetIdFromRef("/path.toml")).toBeNull();
  });

  it("appendUniqueDatasetPaths dedupes by canonical ref", () => {
    const out = appendUniqueDatasetPaths(
      ["renga-flow-dataset:1:foo"],
      ["renga-flow-dataset:1", "renga-flow-dataset:2"]
    );
    expect(out).toEqual(["renga-flow-dataset:1:foo", "renga-flow-dataset:2"]);
  });

  it("non-ref values pass through", () => {
    const p = parseDatasetLibraryRef("examples/minimal_dataset.toml");
    expect(p.isRef).toBe(false);
    expect(p.canonical).toBe("examples/minimal_dataset.toml");
  });
});

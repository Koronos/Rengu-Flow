import { describe, expect, it } from "vitest";
import {
  appendUniqueDatasetPaths,
  canonicalDatasetRef,
  coerceTrainingDatasetEntries,
  datasetRefDisplayLabel,
  formatDatasetLibraryRef,
  isLibraryDatasetRef,
  libraryDatasetIdFromRef,
  parseDatasetLibraryRef,
  trainingDatasetFormValue,
} from "./datasetLibraryRef";

describe("datasetLibraryRef", () => {
  it("parses ref with display suffix", () => {
    const ref = "rengu-flow-dataset:3:artista 1";
    expect(isLibraryDatasetRef(ref)).toBe(true);
    const p = parseDatasetLibraryRef(ref);
    expect(p.id).toBe("3");
    expect(p.label).toBe("artista 1");
    expect(canonicalDatasetRef(ref)).toBe("rengu-flow-dataset:3");
  });

  it("parses ref without suffix", () => {
    const ref = "rengu-flow-dataset:12";
    const p = parseDatasetLibraryRef(ref);
    expect(p.id).toBe("12");
    expect(p.label).toBeNull();
  });

  it("formats ref with optional label", () => {
    expect(formatDatasetLibraryRef(5, "My set")).toBe("rengu-flow-dataset:5:My set");
    expect(formatDatasetLibraryRef(5)).toBe("rengu-flow-dataset:5");
  });

  it("display label prefers suffix then id", () => {
    expect(datasetRefDisplayLabel("rengu-flow-dataset:3:artista 1")).toBe("artista 1");
    expect(datasetRefDisplayLabel("rengu-flow-dataset:12")).toBe("12");
    expect(datasetRefDisplayLabel("/data/foo.toml")).toBe("/data/foo.toml");
  });

  it("libraryDatasetIdFromRef validates numeric id", () => {
    expect(libraryDatasetIdFromRef("rengu-flow-dataset:3:label")).toBe("3");
    expect(libraryDatasetIdFromRef("rengu-flow-dataset:abc")).toBeNull();
    expect(libraryDatasetIdFromRef("/path.toml")).toBeNull();
  });

  it("appendUniqueDatasetPaths dedupes by canonical ref", () => {
    const out = appendUniqueDatasetPaths(
      ["rengu-flow-dataset:1:foo"],
      ["rengu-flow-dataset:1", "rengu-flow-dataset:2"]
    );
    expect(out).toEqual(["rengu-flow-dataset:1:foo", "rengu-flow-dataset:2"]);
  });

  it("non-ref values pass through", () => {
    const p = parseDatasetLibraryRef("examples/minimal_dataset.toml");
    expect(p.isRef).toBe(false);
    expect(p.canonical).toBe("examples/minimal_dataset.toml");
  });

  it("coerceTrainingDatasetEntries keeps arrays", () => {
    const paths = ["rengu-flow-dataset:1:Dataset 1", "rengu-flow-dataset:9:Dataset 9"];
    expect(coerceTrainingDatasetEntries(paths)).toEqual(paths);
  });

  it("coerceTrainingDatasetEntries recovers String(array) merge", () => {
    const merged =
      "rengu-flow-dataset:1:Dataset 1,rengu-flow-dataset:9:Dataset 9";
    expect(coerceTrainingDatasetEntries(merged)).toEqual([
      "rengu-flow-dataset:1:Dataset 1",
      "rengu-flow-dataset:9:Dataset 9",
    ]);
  });

  it("coerceTrainingDatasetEntries leaves single path alone", () => {
    expect(coerceTrainingDatasetEntries("/data/foo.toml")).toEqual(["/data/foo.toml"]);
  });

  it("trainingDatasetFormValue matches backend single-vs-list", () => {
    expect(trainingDatasetFormValue([])).toBe("");
    expect(trainingDatasetFormValue(["a.toml"])).toBe("a.toml");
    expect(trainingDatasetFormValue(["a.toml", "b.toml"])).toEqual(["a.toml", "b.toml"]);
  });
});

import { describe, expect, it } from "vitest";
import {
  datasetRefToThumbSource,
  libraryThumbSource,
  pathThumbSource,
} from "./previewThumbs";

describe("datasetRefToThumbSource", () => {
  it("maps library refs to library thumb source", () => {
    expect(datasetRefToThumbSource("renga-flow-dataset:3:artista 1")).toEqual(
      libraryThumbSource("3")
    );
    expect(datasetRefToThumbSource("renga-flow-dataset:12")).toEqual(libraryThumbSource("12"));
  });

  it("maps non-library paths to path thumb source", () => {
    expect(datasetRefToThumbSource("examples/minimal_dataset.toml")).toEqual(
      pathThumbSource("examples/minimal_dataset.toml")
    );
    expect(datasetRefToThumbSource("/abs/data.toml")).toEqual(
      pathThumbSource("/abs/data.toml")
    );
  });

  it("returns null for empty or invalid library ids", () => {
    expect(datasetRefToThumbSource("")).toBeNull();
    expect(datasetRefToThumbSource("   ")).toBeNull();
    expect(datasetRefToThumbSource("renga-flow-dataset:abc")).toEqual(
      pathThumbSource("renga-flow-dataset:abc")
    );
  });
});

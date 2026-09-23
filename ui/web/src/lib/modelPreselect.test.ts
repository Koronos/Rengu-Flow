import { describe, expect, it } from "vitest";
import { preselectModel, preselectTagModels } from "./modelPreselect";

const model = (id: string, downloaded: boolean) => ({ id, repo_id: id, downloaded, available: true });

describe("preselectTagModels", () => {
  it("takes every downloaded tagger", () => {
    expect(
      preselectTagModels([model("a", false), model("b", true), model("c", true)]),
    ).toEqual(["b", "c"]);
  });

  it("falls back to the registry's first two when none is downloaded", () => {
    expect(
      preselectTagModels([model("a", false), model("b", false), model("c", false)]),
    ).toEqual(["a", "b"]);
  });

  it("is empty for an empty registry", () => {
    expect(preselectTagModels([])).toEqual([]);
  });
});

describe("preselectModel", () => {
  it("takes the registry's first model, downloaded or not", () => {
    expect(preselectModel([model("first", false), model("second", true)])).toBe("first");
  });

  it("is empty for an empty registry", () => {
    expect(preselectModel([])).toBe("");
  });
});

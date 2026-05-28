import { describe, expect, it } from "vitest";
import {
  duplicatePreviewEntry,
  normalizePreviewEntries,
  previewEntryName,
  serializePreviewEntry,
} from "./previewEntries";

describe("previewEntries", () => {
  it("normalizes array values", () => {
    expect(normalizePreviewEntries(["a", "b"])).toEqual(["a", "b"]);
    expect(normalizePreviewEntries(null)).toEqual([]);
  });

  it("serializes simple prompt as string", () => {
    expect(serializePreviewEntry({ prompt: "a cat" })).toBe("a cat");
  });

  it("serializes named table", () => {
    expect(serializePreviewEntry({ name: "portrait", prompt: "1girl" })).toEqual({
      name: "portrait",
      prompt: "1girl",
    });
  });

  it("serializes overrides on table", () => {
    expect(
      serializePreviewEntry({ prompt: "scene", seed: 42, preview_every_n_steps: 100 })
    ).toEqual({
      prompt: "scene",
      seed: 42,
      preview_every_n_steps: 100,
    });
  });

  it("duplicate adds copy suffix to name", () => {
    const dup = duplicatePreviewEntry({ name: "tag", prompt: "x" });
    expect(dup).toEqual({ name: "tag (copy)", prompt: "x" });
  });

  it("labels string entries", () => {
    expect(previewEntryName("short prompt", 0)).toBe("short prompt");
    expect(previewEntryName({ name: "a", prompt: "b" }, 1)).toBe("a");
  });
});

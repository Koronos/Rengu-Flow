import { describe, expect, it } from "vitest";
import {
  captionEditorLocation,
  captionKeyAction,
  isDraftDirty,
  linesFromText,
  parseCaptionEditorQuery,
  stepTarget,
  textFromLines,
} from "./captionEditor";

describe("caption lines", () => {
  it("maps the textarea to lines the way the store writes them", () => {
    expect(linesFromText("  Make it red. \r\n\n  variant 2\n   ")).toEqual(["Make it red.", "variant 2"]);
    expect(linesFromText("")).toEqual([]);
    expect(textFromLines(["a", "b"])).toBe("a\nb");
  });

  it("ignores whitespace-only differences when deciding dirty", () => {
    expect(isDraftDirty(["a", "b"], "a\n\n b \n")).toBe(false);
    expect(isDraftDirty(["a", "b"], "a")).toBe(true);
    expect(isDraftDirty(["a"], "A")).toBe(true);
    expect(isDraftDirty([], "")).toBe(false);
  });
});

describe("stepTarget", () => {
  const page = { offset: 60, limit: 60, total: 150, count: 60 };

  it("stays on the page while it can", () => {
    expect(stepTarget(3, 1, page)).toEqual({ kind: "local", index: 4 });
    expect(stepTarget(3, -1, page)).toEqual({ kind: "local", index: 2 });
  });

  it("crosses to the neighbouring page at the edges", () => {
    expect(stepTarget(59, 1, page)).toEqual({ kind: "page", offset: 120, at: "first" });
    expect(stepTarget(0, -1, page)).toEqual({ kind: "page", offset: 0, at: "last" });
  });

  it("stops at the ends of the listing", () => {
    expect(stepTarget(29, 1, { offset: 120, limit: 60, total: 150, count: 30 })).toBeNull();
    expect(stepTarget(0, -1, { offset: 0, limit: 60, total: 150, count: 60 })).toBeNull();
  });
});

describe("captionKeyAction", () => {
  it("saves on Ctrl/⌘+Enter, even in the textarea", () => {
    expect(captionKeyAction({ key: "Enter", ctrlKey: true, inTextField: true })).toBe("save");
    expect(captionKeyAction({ key: "Enter", metaKey: true })).toBe("save");
    expect(captionKeyAction({ key: "Enter", inTextField: true })).toBeNull();
  });

  it("navigates with Alt+arrows anywhere, plain arrows only outside text fields", () => {
    expect(captionKeyAction({ key: "ArrowDown", altKey: true, inTextField: true })).toBe("next");
    expect(captionKeyAction({ key: "ArrowUp", altKey: true, inTextField: true })).toBe("prev");
    expect(captionKeyAction({ key: "ArrowDown", inTextField: true })).toBeNull();
    expect(captionKeyAction({ key: "ArrowDown" })).toBe("next");
    expect(captionKeyAction({ key: "ArrowUp", ctrlKey: true })).toBeNull();
  });
});

describe("route query", () => {
  it("round-trips a target and omits defaults", () => {
    const loc = captionEditorLocation({ path: "/d/t", control_path: "/d/c", format: "sidecar", ext: ".txt" });
    expect(loc).toEqual({ name: "prep-captions", query: { path: "/d/t", control_path: "/d/c" } });
    const json = captionEditorLocation({ path: "/d/t", format: "json", ext: ".caption" });
    expect(json).toEqual({ name: "prep-captions", query: { path: "/d/t", format: "json" } });
    expect(parseCaptionEditorQuery({ path: "/d/t", ext: ".caption" })).toEqual({
      path: "/d/t",
      control_path: "",
      format: "sidecar",
      ext: ".caption",
    });
    expect(parseCaptionEditorQuery({ path: ["/a", "/b"], format: "bogus" }).format).toBe("sidecar");
  });
});

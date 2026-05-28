import { describe, expect, it } from "vitest";
import { useDatasetImageViewer } from "./useDatasetImageViewer";

describe("useDatasetImageViewer", () => {
  it("opens and closes a single shared viewer", () => {
    const a = useDatasetImageViewer();
    const b = useDatasetImageViewer();

    a.openDatasetImageViewer(["/a.jpg", "/b.jpg"], 1);
    expect(a.viewerOpen.value).toBe(true);
    expect(b.viewerIndex.value).toBe(1);
    expect(b.viewerUrls.value).toEqual(["/a.jpg", "/b.jpg"]);

    b.closeDatasetImageViewer();
    expect(a.viewerOpen.value).toBe(false);
  });

  it("ignores invalid open requests", () => {
    const { viewerOpen, openDatasetImageViewer } = useDatasetImageViewer();
    openDatasetImageViewer([], 0);
    openDatasetImageViewer(["/a.jpg"], -1);
    expect(viewerOpen.value).toBe(false);
  });
});

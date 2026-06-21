import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useDatasetImageViewerStore } from "./datasetImageViewer";

describe("useDatasetImageViewerStore", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it("opens and closes a single shared viewer", () => {
    const a = useDatasetImageViewerStore();
    const b = useDatasetImageViewerStore();

    a.openDatasetImageViewer(["/a.jpg", "/b.jpg"], 1);
    expect(a.viewerOpen).toBe(true);
    expect(b.viewerIndex).toBe(1);
    expect(b.viewerUrls).toEqual(["/a.jpg", "/b.jpg"]);

    b.closeDatasetImageViewer();
    expect(a.viewerOpen).toBe(false);
  });

  it("ignores invalid open requests", () => {
    const store = useDatasetImageViewerStore();
    store.openDatasetImageViewer([], 0);
    store.openDatasetImageViewer(["/a.jpg"], -1);
    expect(store.viewerOpen).toBe(false);
  });
});

/**
 * The caption editor's safety contract: leaving an image with unsaved text saves it first, a
 * failed save keeps you (and your text) where you are, and a read-only folder is never written.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const { prepCaptions, prepSaveCaption } = vi.hoisted(() => ({
  prepCaptions: vi.fn(),
  prepSaveCaption: vi.fn(),
}));
vi.mock("../api", () => ({ api: { prepCaptions, prepSaveCaption } }));

import type { CaptionListParams, CaptionListResult, CaptionSaveBody } from "../types/api";
import { useCaptionEditor } from "./useCaptionEditor";

const TARGET = { path: "/d/targets", control_path: "/d/controls", format: "sidecar" as const, ext: ".txt" };

/** A folder of `total` images named img<N>.png, served `limit` at a time. */
function fakeFolder(total: number, { readOnly = false } = {}) {
  const lines: Record<string, string[]> = {};
  for (let i = 0; i < total; i++) lines[`img${i}.png`] = [`caption ${i}`];
  prepCaptions.mockImplementation(async (p: CaptionListParams): Promise<CaptionListResult> => {
    let keys = Object.keys(lines);
    if (p.q) keys = keys.filter((k) => lines[k].join(" ").includes(p.q!));
    const offset = p.offset ?? 0;
    const limit = p.limit ?? 60;
    return {
      path: p.path,
      format: "sidecar",
      ext: ".txt",
      control_path: p.control_path ?? null,
      image_count: total,
      uncaptioned_count: 0,
      unpaired_count: 0,
      total: keys.length,
      offset,
      limit,
      read_only: readOnly,
      active: [],
      items: keys.slice(offset, offset + limit).map((key) => ({
        key,
        lines: [...lines[key]],
        token: `t-${key}`,
        controls: [],
        unpaired: null,
      })),
    };
  });
  prepSaveCaption.mockImplementation(async (b: CaptionSaveBody) => {
    lines[b.key] = b.lines;
    return { key: b.key, lines: b.lines, written: [b.key], backup: b.backup ? "20260923T000000Z" : null };
  });
  return lines;
}

beforeEach(() => {
  prepCaptions.mockReset();
  prepSaveCaption.mockReset();
});

describe("useCaptionEditor", () => {
  it("opens on the first image with its caption in the draft", async () => {
    fakeFolder(3);
    const ed = useCaptionEditor(2);
    expect(await ed.open(TARGET)).toBe(true);
    expect(prepCaptions).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/d/targets", control_path: "/d/controls", limit: 2, offset: 0 })
    );
    expect(ed.current.value?.key).toBe("img0.png");
    expect(ed.draft.value).toBe("caption 0");
    expect(ed.dirty.value).toBe(false);
  });

  it("autosaves a dirty draft when selecting another image, with a backup only the first time", async () => {
    const disk = fakeFolder(3);
    const ed = useCaptionEditor();
    await ed.open(TARGET);

    ed.draft.value = "Make it red.\nsecond line";
    expect(ed.dirty.value).toBe(true);
    expect(await ed.select(1)).toBe(true);
    expect(prepSaveCaption).toHaveBeenCalledTimes(1);
    expect(prepSaveCaption.mock.calls[0][0]).toMatchObject({
      key: "img0.png",
      lines: ["Make it red.", "second line"],
      expected: ["caption 0"],
      backup: true,
    });
    expect(disk["img0.png"]).toEqual(["Make it red.", "second line"]);
    expect(ed.items.value[0].lines).toEqual(["Make it red.", "second line"]);
    expect(ed.backupName.value).toBe("20260923T000000Z");
    expect(ed.draft.value).toBe("caption 1");

    ed.draft.value = "fixed 1";
    await ed.select(2);
    expect(prepSaveCaption.mock.calls[1][0]).toMatchObject({ key: "img1.png", backup: false });
  });

  it("does not save when the draft is unchanged", async () => {
    fakeFolder(3);
    const ed = useCaptionEditor();
    await ed.open(TARGET);
    ed.draft.value = "caption 0\n\n";
    await ed.select(1);
    await ed.move(1);
    expect(prepSaveCaption).not.toHaveBeenCalled();
    expect(ed.selected.value).toBe(2);
  });

  it("stays on the image, keeping the text, when the autosave fails", async () => {
    fakeFolder(3);
    const ed = useCaptionEditor();
    await ed.open(TARGET);
    prepSaveCaption.mockRejectedValueOnce(new Error("The caption of img0.png changed on disk"));

    ed.draft.value = "my careful fix";
    expect(await ed.move(1)).toBe(false);
    expect(ed.selected.value).toBe(0);
    expect(ed.draft.value).toBe("my careful fix");
    expect(ed.dirty.value).toBe(true);
    expect(ed.error.value).toContain("changed on disk");

    // reload discards the draft for what is on disk.
    await ed.reload();
    expect(ed.draft.value).toBe("caption 0");
    expect(ed.selected.value).toBe(0);
  });

  it("walks across pages with next/previous, autosaving on the way", async () => {
    fakeFolder(5);
    const ed = useCaptionEditor(2);
    await ed.open(TARGET);
    await ed.move(1);
    ed.draft.value = "edited 1";
    expect(await ed.move(1)).toBe(true); // off the end of page 1
    expect(prepSaveCaption.mock.calls[0][0]).toMatchObject({ key: "img1.png", lines: ["edited 1"] });
    expect(prepCaptions).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 2 }));
    expect(ed.current.value?.key).toBe("img2.png");

    expect(await ed.move(-1)).toBe(true); // back onto the last item of page 1
    expect(ed.current.value?.key).toBe("img1.png");
    expect(ed.draft.value).toBe("edited 1");

    await ed.goToPage(3);
    expect(ed.current.value?.key).toBe("img4.png");
    expect(await ed.move(1)).toBe(false); // end of the listing
    expect(ed.current.value?.key).toBe("img4.png");
  });

  it("re-queries with the search and filter from the first page, saving first", async () => {
    fakeFolder(12);
    const ed = useCaptionEditor(5);
    await ed.goToPage(1); // no folder yet: nothing happens
    expect(prepCaptions).not.toHaveBeenCalled();
    await ed.open(TARGET);
    await ed.goToPage(2);
    ed.draft.value = "caption 5 fixed";
    ed.search.value = " caption 1 ";
    ed.filter.value = "uncaptioned";
    await ed.applyQuery();
    expect(prepSaveCaption).toHaveBeenCalledTimes(1);
    expect(prepCaptions).toHaveBeenLastCalledWith(
      expect.objectContaining({ q: "caption 1", filter: "uncaptioned", offset: 0 })
    );
    expect(ed.items.value.map((i) => i.key)).toEqual(["img1.png", "img10.png", "img11.png"]);
  });

  it("never writes a read-only folder", async () => {
    fakeFolder(3, { readOnly: true });
    const ed = useCaptionEditor();
    await ed.open(TARGET);
    ed.draft.value = "racing the job";
    expect(await ed.save()).toBe(false);
    expect(await ed.select(1)).toBe(false);
    expect(prepSaveCaption).not.toHaveBeenCalled();
    expect(ed.selected.value).toBe(0);
    expect(ed.error.value).toMatch(/read-only/);
  });

  it("asks for a new backup after opening another folder", async () => {
    fakeFolder(2);
    const ed = useCaptionEditor();
    await ed.open(TARGET);
    ed.draft.value = "x";
    await ed.save();
    await ed.open({ ...TARGET, path: "/d/other" });
    expect(ed.backupName.value).toBeNull();
    ed.draft.value = "y";
    await ed.save();
    expect(prepSaveCaption.mock.calls.map((c) => c[0].backup)).toEqual([true, true]);
    expect(prepSaveCaption.mock.calls[1][0].path).toBe("/d/other");
  });
});

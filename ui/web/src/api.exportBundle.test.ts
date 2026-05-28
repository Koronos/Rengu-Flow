import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

describe("api.exportConfigBundle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reads ZIP body once on success", async () => {
    const zipBytes = new Uint8Array([0x50, 0x4b, 0x03, 0x04]);
    const blobSpy = vi.spyOn(Response.prototype, "blob");
    const textSpy = vi.spyOn(Response.prototype, "text");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(zipBytes, {
          status: 200,
          headers: {
            "Content-Disposition": 'attachment; filename="my_run.zip"',
          },
        })
      )
    );

    const { blob, filename } = await api.exportConfigBundle("my_run", 'dataset = "renga-flow-dataset:9"\n');

    expect(blobSpy).toHaveBeenCalledOnce();
    expect(textSpy).not.toHaveBeenCalled();
    expect(blob.size).toBe(zipBytes.length);
    expect(filename).toBe("my_run.zip");
  });

  it("parses JSON error without calling blob on failure", async () => {
    const blobSpy = vi.spyOn(Response.prototype, "blob");
    const textSpy = vi.spyOn(Response.prototype, "text");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ detail: "Dataset not found" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        })
      )
    );

    await expect(
      api.exportConfigBundle("missing", 'dataset = "renga-flow-dataset:9"\n')
    ).rejects.toThrow("Dataset not found");

    expect(textSpy).toHaveBeenCalledOnce();
    expect(blobSpy).not.toHaveBeenCalled();
  });
});

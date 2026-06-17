import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
});

const SAMPLE = {
  path: "/repo/rengu.local.toml",
  exists: true,
  editable: {
    training: { num_gpus: 1, master_port: 29500, extra_args: "", env: {} },
    maintenance: { enabled: false, allow_pip: false },
  },
  restartRequired: { ui: { public: false, token: null } },
  readOnly: { ui: { host: "127.0.0.1", port: 8765, data_dir: "data" } },
};

describe("settings api", () => {
  it("getSettings fetches /settings", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(SAMPLE), { status: 200 }));
    const out = await api.getSettings();
    expect(out.editable.training.num_gpus).toBe(1);
    expect(fetchMock).toHaveBeenCalledWith("/api/v1/settings", expect.anything());
  });

  it("updateSettings PUTs the patch", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(SAMPLE), { status: 200 }));
    await api.updateSettings({ training: { num_gpus: 2 } });
    const [, opts] = fetchMock.mock.calls[0];
    expect(opts?.method).toBe("PUT");
    expect(JSON.parse(opts?.body as string)).toEqual({ training: { num_gpus: 2 } });
  });
});

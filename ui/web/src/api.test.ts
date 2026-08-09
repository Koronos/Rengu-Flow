/**
 * `request<T>`'s error contract.
 *
 * Two properties are load-bearing and neither is local to `api.ts`:
 *
 * 1. **The status survives.** `useWorkflowEditor` distinguishes a 409 (another tab saved first, or
 *    the run started) from every other failure, and a lost `status` would turn the one
 *    unrecoverable case in the app into a generic red toast.
 * 2. **It is still an `Error`.** Every other caller in the repo reads `e instanceof Error` /
 *    `e.message` (`lib/formatError`), so `ApiError` may add to that contract but never replace it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./api";

interface FakeResponse {
  ok: boolean;
  status: number;
  statusText: string;
  text: () => Promise<string>;
}

function respond(status: number, body: string, statusText = ""): FakeResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText,
    text: async () => body,
  };
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  (globalThis as unknown as { fetch: unknown }).fetch = fetchMock;
});

afterEach(() => {
  vi.restoreAllMocks();
});

/** Run any `request`-backed call and hand back whatever it threw. */
async function thrown(status: number, body: string, statusText = ""): Promise<unknown> {
  fetchMock.mockResolvedValue(respond(status, body, statusText));
  try {
    await api.getWorkflow(7);
    return null;
  } catch (e) {
    return e;
  }
}

describe("request error handling", () => {
  it("throws an ApiError carrying the response status", async () => {
    const error = await thrown(409, JSON.stringify({ detail: "Workflow changed" }));

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(409);
    expect((error as ApiError).message).toBe("Workflow changed");
  });

  it("reports each status distinctly, so callers can branch on 409 alone", async () => {
    const conflict = (await thrown(409, '{"detail":"nope"}')) as ApiError;
    const notFound = (await thrown(404, '{"detail":"nope"}')) as ApiError;
    const server = (await thrown(500, '{"detail":"nope"}')) as ApiError;

    expect([conflict.status, notFound.status, server.status]).toEqual([409, 404, 500]);
  });

  it("stays an Error: every other caller in the repo reads instanceof Error / .message", async () => {
    const error = await thrown(500, JSON.stringify({ detail: "Boom" }));

    expect(error).toBeInstanceOf(Error);
    expect(error instanceof Error && error.message).toBe("Boom");
    expect((error as Error).name).toBe("ApiError");
    expect(String(error)).toContain("Boom");
  });

  it("uses a non-JSON body as the message rather than losing it", async () => {
    const error = (await thrown(502, "<html>Bad Gateway</html>")) as ApiError;

    expect(error.status).toBe(502);
    expect(error.message).toBe("<html>Bad Gateway</html>");
  });

  it("falls back to the status line, and then to HTTP <status>, never to an empty message", async () => {
    const withStatusText = (await thrown(503, "", "Service Unavailable")) as ApiError;
    expect(withStatusText.message).toBe("Service Unavailable");

    const bare = (await thrown(503, "")) as ApiError;
    expect(bare.message).toBe("HTTP 503");
  });

  it("returns the decoded body on success and never throws", async () => {
    fetchMock.mockResolvedValue(respond(200, JSON.stringify({ id: 7, version: 3 })));

    await expect(api.getWorkflow(7)).resolves.toEqual({ id: 7, version: 3 });
  });
});

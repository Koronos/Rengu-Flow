/** Turn API / fetch errors into a readable string (never "[object Object]"). */

export function formatApiDetail(detail: unknown): string {
  if (detail == null || detail === "") return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const rec = item as Record<string, unknown>;
          const loc = Array.isArray(rec.loc)
            ? rec.loc.filter(Boolean).join(".")
            : "";
          const msg = rec.msg || rec.message || "";
          if (loc && msg) return `${loc}: ${msg}`;
          if (msg) return String(msg);
        }
        try {
          return JSON.stringify(item);
        } catch {
          return String(item);
        }
      })
      .filter(Boolean)
      .join("; ");
  }
  if (typeof detail === "object") {
    const rec = detail as Record<string, unknown>;
    if (typeof rec.message === "string") return rec.message;
    if (typeof rec.error === "string") return rec.error;
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }
  return String(detail);
}

export function errorMessageFromResponseBody(
  data: unknown,
  fallbackStatus = ""
): string {
  const body = data as Record<string, unknown> | null;
  return (
    formatApiDetail(body?.detail) ||
    (typeof body?.error === "string" ? body.error : formatApiDetail(body?.error)) ||
    fallbackStatus
  );
}

export function formatError(err: unknown): string {
  if (err == null) return "Unknown error";
  if (typeof err === "string") return err;
  if (err instanceof Error) {
    const msg = err.message?.trim();
    if (msg && msg !== "[object Object]") return msg;
  }
  if (typeof err === "object") {
    const rec = err as Record<string, unknown>;
    const fromDetail = formatApiDetail(rec.detail ?? err);
    if (fromDetail) return fromDetail;
  }
  const fallback = String(err);
  return fallback === "[object Object]" ? "Request failed" : fallback;
}

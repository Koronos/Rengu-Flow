/** Turn API / fetch errors into a readable string (never "[object Object]"). */

export function formatApiDetail(detail) {
  if (detail == null || detail === "") return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const loc = Array.isArray(item.loc) ? item.loc.filter(Boolean).join(".") : "";
          const msg = item.msg || item.message || "";
          if (loc && msg) return `${loc}: ${msg}`;
          if (msg) return msg;
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
    if (typeof detail.message === "string") return detail.message;
    if (typeof detail.error === "string") return detail.error;
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }
  return String(detail);
}

export function formatError(err) {
  if (err == null) return "Unknown error";
  if (typeof err === "string") return err;
  if (err instanceof Error) {
    const msg = err.message?.trim();
    if (msg && msg !== "[object Object]") return msg;
  }
  if (typeof err === "object") {
    const fromDetail = formatApiDetail(err.detail ?? err);
    if (fromDetail) return fromDetail;
  }
  const fallback = String(err);
  return fallback === "[object Object]" ? "Request failed" : fallback;
}

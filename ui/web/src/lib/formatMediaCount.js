/** Format image/video counts; append + when scan was capped. */
export function formatMediaCount(count, capped) {
  if (count == null || count === "") return "—";
  const n = Number(count);
  if (Number.isNaN(n)) return String(count);
  if (capped) return `${n.toLocaleString()}+`;
  return n.toLocaleString();
}

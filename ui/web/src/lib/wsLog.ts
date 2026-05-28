/** Shared WebSocket URL and bounded in-memory log tail helpers. */

export const MAX_LOG_CHARS = 512_000;

export function wsBaseUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

export function trimBoundedLogChunks(chunks: string[], charCount: { value: number }): void {
  while (charCount.value > MAX_LOG_CHARS && chunks.length > 1) {
    const removed = chunks.shift();
    if (removed) charCount.value -= removed.length;
  }
}

export function appendBoundedLogChunk(
  chunks: string[],
  charCount: { value: number },
  text: string
): void {
  if (!text) return;
  chunks.push(text);
  charCount.value += text.length;
  trimBoundedLogChunks(chunks, charCount);
}

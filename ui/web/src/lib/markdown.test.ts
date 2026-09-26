import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./markdown";

function parse(html: string): HTMLDivElement {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div;
}

// Every fixture keeps a leading paragraph before the fenced block, matching real docs
// (e.g. docs/user/training-qwen-image21.md): the toolbar/pre are never the sole/first
// top-level node of the sanitized fragment.

describe("renderMarkdown", () => {
  it("wraps a toml fenced block in an Open/Copy toolbar", () => {
    const md = 'Some intro text.\n\n```toml\nkey = "value"\n```\n';
    const html = renderMarkdown(md);
    const el = parse(html);

    const block = el.querySelector(".md-code-block");
    expect(block).not.toBeNull();

    const openBtn = el.querySelector(".md-snippet-open");
    const copyBtn = el.querySelector(".md-snippet-copy");
    expect(openBtn).not.toBeNull();
    expect(copyBtn).not.toBeNull();
    expect(openBtn!.getAttribute("data-snippet-lang")).toBe("toml");

    // the visible <pre><code> preview is preserved, escaped as before
    const code = el.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code!.className).toBe("language-toml");
    expect(code!.textContent).toBe('key = "value"\n');
  });

  it("does not add a toolbar for other languages", () => {
    const md = "Some intro text.\n\n```bash\necho hi\n```\n";
    const html = renderMarkdown(md);
    const el = parse(html);

    expect(el.querySelector(".md-code-block")).toBeNull();
    expect(el.querySelector(".md-snippet-open")).toBeNull();

    const code = el.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code!.className).toBe("language-bash");
    expect(code!.textContent).toBe("echo hi\n");
  });

  it("renders plain code (no lang) exactly as before, without a toolbar", () => {
    const md = "Some intro text.\n\n```\nplain text\n```\n";
    const html = renderMarkdown(md);
    const el = parse(html);

    expect(el.querySelector(".md-code-block")).toBeNull();
    const code = el.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code!.textContent).toBe("plain text\n");
  });

  it("round-trips the raw snippet through data-snippet exactly (quotes, <, &, newlines)", () => {
    const raw = 'title = "a<b> & c\'s"\nother = 1\n\nmore = "x"';
    const md = "Some intro text.\n\n```toml\n" + raw + "\n```\n";
    const html = renderMarkdown(md);
    const el = parse(html);

    const openBtn = el.querySelector(".md-snippet-open")!;
    const copyBtn = el.querySelector(".md-snippet-copy")!;
    expect(openBtn.getAttribute("data-snippet")).toBe(raw);
    expect(copyBtn.getAttribute("data-snippet")).toBe(raw);
  });
});

/** GitHub-flavored markdown → safe HTML for in-app documentation. */

import DOMPurify from "dompurify";
import { marked } from "marked";

const SANITIZE_OPTS = {
  ADD_ATTR: ["target", "rel", "data-doc-path", "class"],
  ADD_TAGS: ["table", "thead", "tbody", "tr", "th", "td"],
};

/** Fenced code languages that get an "Open in modal" + "Copy" toolbar instead of a plain <pre>. */
export const SNIPPET_MODAL_LANGS = new Set(["toml"]);

const parseContext = { docPath: "" };

marked.setOptions({ gfm: true, breaks: false });

marked.use({
  renderer: {
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens);
      const docTarget = resolveDocLink(href, parseContext.docPath);
      const titleAttr = title ? ` title="${escapeAttr(title)}"` : "";
      if (docTarget) {
        return `<a href="#" class="md-doc-link" data-doc-path="${escapeAttr(docTarget)}"${titleAttr}>${text}</a>`;
      }
      const safeHref = escapeAttr(href || "#");
      return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer"${titleAttr}>${text}</a>`;
    },
    code({ text, lang }) {
      const language = (lang || "").trim().toLowerCase();
      if (!SNIPPET_MODAL_LANGS.has(language)) {
        return false;
      }
      const escapedLang = escapeAttr(language);
      const escaped = escapeAttr(text);
      return (
        `<div class="md-code-block">` +
        `<div class="md-code-toolbar">` +
        `<span class="md-code-lang">${escapedLang}</span>` +
        `<button type="button" class="md-code-btn md-snippet-open" data-snippet="${escaped}" data-snippet-lang="${escapedLang}">Open</button>` +
        `<button type="button" class="md-code-btn md-snippet-copy" data-snippet="${escaped}">Copy</button>` +
        `</div>` +
        `<pre><code class="language-${escapedLang}">${escaped}\n</code></pre>` +
        `</div>`
      );
    },
  },
});

/**
 * Resolve a markdown link target to a docs/ path served by the API.
 * @param {string} href
 * @param {string} baseDocPath e.g. docs/user/web-ui.md
 * @returns {string|null}
 */
export function resolveDocLink(href: string | null | undefined, baseDocPath: string): string | null {
  if (!href || /^https?:\/\//i.test(href) || /^mailto:/i.test(href)) {
    return null;
  }
  const withoutHash = href.split("#")[0];
  if (!withoutHash.endsWith(".md")) {
    return null;
  }

  if (withoutHash.startsWith("docs/")) {
    return withoutHash;
  }

  const baseParts = baseDocPath.replace(/\\/g, "/").split("/");
  baseParts.pop();
  const linkParts = withoutHash.split("/");
  const out = [...baseParts];

  for (const part of linkParts) {
    if (part === "" || part === ".") continue;
    if (part === "..") {
      out.pop();
    } else {
      out.push(part);
    }
  }

  const resolved = out.join("/");
  if (!resolved.startsWith("docs/")) {
    if (resolved.startsWith("user/") || resolved.startsWith("developer/")) {
      return `docs/${resolved}`;
    }
    return null;
  }
  return resolved;
}

/**
 * @param {string} md
 * @param {{ docPath?: string }} [ctx]
 * @returns {string}
 */
export function renderMarkdown(md: string, ctx: { docPath?: string } = {}): string {
  if (!md) return "";
  parseContext.docPath = ctx.docPath || "";
  const raw = marked.parse(md, { async: false });
  return DOMPurify.sanitize(raw, SANITIZE_OPTS);
}

function escapeAttr(s: unknown): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

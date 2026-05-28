/** GitHub-flavored markdown → safe HTML for in-app documentation. */

import DOMPurify from "dompurify";
import { marked } from "marked";

const SANITIZE_OPTS = {
  ADD_ATTR: ["target", "rel", "data-doc-path", "class"],
  ADD_TAGS: ["table", "thead", "tbody", "tr", "th", "td"],
};

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

import { ref } from "vue";
import { ElMessage } from "element-plus";
import { copyText } from "../lib/clipboard";

/**
 * Shared "open snippet in a modal" + "copy" behavior for the .md-code-block toolbar
 * that lib/markdown.ts renders around fenced code blocks (e.g. toml).
 */
export function useCodeSnippetDialog() {
  const dialogVisible = ref(false);
  const dialogTitle = ref("");
  const dialogContent = ref("");

  function openSnippet(text: string, lang: string): void {
    dialogContent.value = text;
    dialogTitle.value = lang ? lang.toUpperCase() : "Snippet";
    dialogVisible.value = true;
  }

  async function copySnippet(text: string): Promise<void> {
    const ok = await copyText(text);
    if (ok) {
      ElMessage.success("Copied to clipboard");
    } else {
      ElMessage.error("Copy failed");
    }
  }

  /** Handles a click inside a rendered .md-article; returns true if it consumed the event. */
  function onCodeSnippetClick(event: MouseEvent): boolean {
    const target = event.target;
    if (!(target instanceof Element)) return false;

    const openBtn = target.closest(".md-snippet-open");
    if (openBtn) {
      openSnippet(openBtn.getAttribute("data-snippet") || "", openBtn.getAttribute("data-snippet-lang") || "");
      return true;
    }

    const copyBtn = target.closest(".md-snippet-copy");
    if (copyBtn) {
      void copySnippet(copyBtn.getAttribute("data-snippet") || "");
      return true;
    }

    return false;
  }

  return { dialogVisible, dialogTitle, dialogContent, onCodeSnippetClick };
}

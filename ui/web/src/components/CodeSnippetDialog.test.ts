import { afterEach, describe, expect, it, vi } from "vitest";
import { createApp, h, nextTick } from "vue";
import ElementPlus, { ElMessage } from "element-plus";
import CodeSnippetDialog from "./CodeSnippetDialog.vue";

async function flush() {
  await nextTick();
  await Promise.resolve();
  await nextTick();
}

function mountDialog(content: string) {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({
    render: () =>
      h(CodeSnippetDialog, { modelValue: true, title: "TOML", content }),
  });
  app.use(ElementPlus);
  app.mount(el);
  return { el, unmount: () => (app.unmount(), el.remove()) };
}

function findButton(el: HTMLElement, exact: string): HTMLButtonElement | null {
  const buttons = Array.from(el.querySelectorAll("button"));
  return (
    (buttons.find((b) => (b.textContent ?? "").replace(/\s+/g, " ").trim() === exact) as
      | HTMLButtonElement
      | undefined) ?? null
  );
}

describe("CodeSnippetDialog", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("shows the full content and copies the exact text via the clipboard API", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    const successSpy = vi.spyOn(ElMessage, "success").mockImplementation((() => ({})) as never);

    const content = 'key = "value"\nother = 1\n\nmore = "a<b>&c"';
    const { unmount } = mountDialog(content);
    await flush();

    // el-dialog teleports its content to <body>, not the mounted container
    expect(document.body.querySelector(".code-snippet-dialog pre")?.textContent).toBe(content);

    findButton(document.body, "Copy")!.click();
    await flush();

    expect(writeText).toHaveBeenCalledWith(content);
    expect(successSpy).toHaveBeenCalled();
    unmount();
  });

  it("falls back to execCommand when clipboard is unavailable and still reports success", async () => {
    vi.stubGlobal("navigator", {});
    const execCommand = vi.fn().mockReturnValue(true);
    document.execCommand = execCommand;
    const successSpy = vi.spyOn(ElMessage, "success").mockImplementation((() => ({})) as never);

    const content = "fallback content";
    const { unmount } = mountDialog(content);
    await flush();

    findButton(document.body, "Copy")!.click();
    await flush();

    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(successSpy).toHaveBeenCalled();
    unmount();
  });

  it("reports an error when both the clipboard API and execCommand fail", async () => {
    vi.stubGlobal("navigator", {});
    document.execCommand = vi.fn().mockReturnValue(false);
    const errorSpy = vi.spyOn(ElMessage, "error").mockImplementation((() => ({})) as never);

    const { unmount } = mountDialog("content");
    await flush();

    findButton(document.body, "Copy")!.click();
    await flush();

    expect(errorSpy).toHaveBeenCalled();
    unmount();
  });
});

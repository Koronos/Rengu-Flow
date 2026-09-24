import { describe, expect, it } from "vitest";
import { createApp, h, nextTick } from "vue";
import ElementPlus from "element-plus";
import TagFrequencyTable from "./TagFrequencyTable.vue";

interface TagRow {
  tag: string;
  count: number;
}

function tags(n: number): TagRow[] {
  return Array.from({ length: n }, (_, i) => ({ tag: `tag ${i}`, count: n - i }));
}

async function mountTable(initialTags: TagRow[], scope = "line1") {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const events: Record<string, unknown[][]> = {};

  const app = createApp({
    setup() {
      return () =>
        h(TagFrequencyTable, {
          tags: initialTags,
          scope,
          onSelectTag: (...args: unknown[]) => (events["select-tag"] ??= []).push(args),
          onRemoveTag: (...args: unknown[]) => (events["remove-tag"] ??= []).push(args),
          onRenameTag: (...args: unknown[]) => (events["rename-tag"] ??= []).push(args),
          onPrune: (...args: unknown[]) => (events["prune"] ??= []).push(args),
          onScopeChange: (...args: unknown[]) => (events["scope-change"] ??= []).push(args),
        });
    },
  });
  app.use(ElementPlus);
  app.mount(el);
  await nextTick();
  await nextTick();
  await nextTick();

  return {
    el,
    events,
    async unmount() {
      app.unmount();
      el.remove();
    },
  };
}

describe("TagFrequencyTable", () => {
  it("emits select-tag when a row is clicked", async () => {
    const { el, events, unmount } = await mountTable(tags(5));
    const cell = el.querySelector(".tag-freq__cell-tag") as HTMLElement | null;
    expect(cell).toBeTruthy();

    cell!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();

    expect(events["select-tag"]).toEqual([["tag 0"]]);

    await unmount();
  });

  it("emits remove-tag without triggering select-tag when the remove button is clicked", async () => {
    const { el, events, unmount } = await mountTable(tags(5));
    const removeBtn = el.querySelector(
      ".tag-freq__actions button.el-button--danger"
    ) as HTMLElement | null;
    expect(removeBtn).toBeTruthy();

    removeBtn!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();

    expect(events["remove-tag"]).toEqual([["tag 0"]]);
    expect(events["select-tag"]).toBeUndefined();

    await unmount();
  });

  it("sorts by count when the # header is clicked, toggling asc/desc", async () => {
    // counts: tag 0 -> 5, tag 1 -> 4, ..., tag 4 -> 1
    const { el, unmount } = await mountTable(tags(5));
    const header = el.querySelector(
      '.el-table-v2__header-cell[data-key="count"]'
    ) as HTMLElement | null;
    expect(header).toBeTruthy();

    header!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();
    let firstCell = el.querySelector(".tag-freq__cell-tag") as HTMLElement | null;
    expect(firstCell?.textContent).toBe("tag 4"); // ascending: smallest count (1) first

    header!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();
    firstCell = el.querySelector(".tag-freq__cell-tag") as HTMLElement | null;
    expect(firstCell?.textContent).toBe("tag 0"); // descending: largest count (5) first

    await unmount();
  });

  it("filters by substring on the tag name", async () => {
    const { el, unmount } = await mountTable([
      { tag: "red car", count: 3 },
      { tag: "blue sky", count: 1 },
      { tag: "red bike", count: 2 },
    ]);
    const input = el.querySelector(".tag-freq__toolbar input") as HTMLInputElement | null;
    expect(input).toBeTruthy();

    input!.value = "red";
    input!.dispatchEvent(new InputEvent("input", { bubbles: true }));
    await nextTick();
    await nextTick();

    const cells = [...el.querySelectorAll(".tag-freq__cell-tag")].map((n) => n.textContent);
    expect(cells).toEqual(["red car", "red bike"]);

    await unmount();
  });

  it("keeps DOM node count bounded for a 20k tag vocabulary", async () => {
    const t0 = performance.now();
    const { el, unmount } = await mountTable(tags(20_000));
    const mountMs = performance.now() - t0;

    const rowCount = el.querySelectorAll(".el-table-v2__row").length;
    const nodeCount = el.querySelectorAll("*").length;

    // The whole point of virtualizing: rendered DOM stays bounded to roughly
    // what fits the viewport, regardless of vocabulary size (20k here).
    expect(rowCount).toBeLessThan(100);
    expect(nodeCount).toBeLessThan(2000);
    expect(mountMs).toBeLessThan(3000);

    await unmount();
  });
});

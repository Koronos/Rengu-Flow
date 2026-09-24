import { describe, expect, it } from "vitest";
import { createApp, h, nextTick, ref } from "vue";
import ElementPlus from "element-plus";
import TagFilterBuilder from "./TagFilterBuilder.vue";
import type { TagEditOpDto } from "../types/api";

function tagOptions(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `tag ${i}`);
}

async function mountBuilder(initialTagOptions: string[]) {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const model = ref<TagEditOpDto["filter"]>({ all: [], any: [], none: [] });
  const updates: TagEditOpDto["filter"][] = [];

  const app = createApp({
    setup() {
      return () =>
        h(TagFilterBuilder, {
          modelValue: model.value,
          tagOptions: initialTagOptions,
          "onUpdate:modelValue": (value: TagEditOpDto["filter"]) => {
            updates.push(value);
            model.value = value;
          },
        });
    },
  });
  app.use(ElementPlus);
  app.mount(el);
  await nextTick();
  await nextTick();

  return {
    el,
    model,
    updates,
    async unmount() {
      app.unmount();
      el.remove();
    },
  };
}

describe("TagFilterBuilder", () => {
  it("selecting a tag in 'Has all' emits the updated filter", async () => {
    const { el, updates, unmount } = await mountBuilder(tagOptions(5));

    const selects = el.querySelectorAll(".tag-filter-builder__row .el-select");
    expect(selects.length).toBe(3);

    const input = selects[0].querySelector("input") as HTMLInputElement | null;
    expect(input).toBeTruthy();
    input!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();

    const listId = input!.getAttribute("aria-controls");
    const list = document.getElementById(listId ?? "");
    expect(list).toBeTruthy();
    const option = list!.querySelector(".el-select-dropdown__item") as HTMLElement | null;
    expect(option).toBeTruthy();
    option!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();

    expect(updates.at(-1)).toEqual({ all: ["tag 0"], any: [], none: [] });

    await unmount();
  });

  it("filters options by substring", async () => {
    const { el, unmount } = await mountBuilder(["red car", "blue sky", "red bike"]);
    const select = el.querySelector(".tag-filter-builder__row .el-select") as HTMLElement;
    const input = select.querySelector("input") as HTMLInputElement;

    input.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await nextTick();
    input.value = "red";
    input.dispatchEvent(new InputEvent("input", { bubbles: true }));
    await nextTick();
    await nextTick();

    // Each of the three (all/any/none) selects keeps its own persistent
    // dropdown list mounted, so scope the query to this select's own list
    // (linked via aria-controls) instead of the whole document.
    const listId = input.getAttribute("aria-controls");
    const list = document.getElementById(listId ?? "");
    expect(list).toBeTruthy();
    const optionLabels = [...list!.querySelectorAll(".el-select-dropdown__item")].map((n) =>
      n.textContent?.trim()
    );
    // "red" itself is the allow-create row (typed text doesn't match an
    // existing tag verbatim); the substring-matched tags follow it.
    expect(optionLabels).toEqual(["red", "red car", "red bike"]);

    await unmount();
  });

  it("keeps DOM node count bounded for a 20k tag vocabulary", async () => {
    const t0 = performance.now();
    const { el, unmount } = await mountBuilder(tagOptions(20_000));
    const mountMs = performance.now() - t0;

    const nodeCount = el.querySelectorAll("*").length;
    expect(nodeCount).toBeLessThan(2000);
    expect(mountMs).toBeLessThan(3000);

    await unmount();
  });
});

import { describe, expect, it } from "vitest";
import { createApp, nextTick, ref } from "vue";
import ElementPlus from "element-plus";
import ResolutionScheduleField from "./ResolutionScheduleField.vue";

function findButton(el: HTMLElement, text: string): HTMLButtonElement | null {
  const buttons = Array.from(el.querySelectorAll("button"));
  return (buttons.find((b) => (b.textContent ?? "").includes(text)) as HTMLButtonElement) ?? null;
}

describe("ResolutionScheduleField", () => {
  async function mountField(initial: unknown, resolutions: number[]) {
    const model = ref(initial);
    const el = document.createElement("div");
    document.body.appendChild(el);
    const app = createApp({
      components: { ResolutionScheduleField },
      setup() {
        return { model, resolutions };
      },
      template:
        '<ResolutionScheduleField v-model="model" :available-resolutions="resolutions" />',
    });
    app.use(ElementPlus);
    app.mount(el);
    await nextTick();
    return {
      el,
      model,
      async unmount() {
        app.unmount();
        el.remove();
      },
    };
  }

  it("adds a stage prefilled with the first resolution and fraction 1", async () => {
    const { el, model, unmount } = await mountField("", [512, 768, 1024]);
    const addBtn = findButton(el, "Add stage");
    expect(addBtn).toBeTruthy();
    addBtn!.click();
    await nextTick();
    expect(JSON.parse(model.value as string)).toEqual({
      enabled: true,
      stage: [{ resolutions: [512], fraction: 1 }],
    });
    await unmount();
  });

  it("shows a hint and no add button when there are no resolutions", async () => {
    const { el, unmount } = await mountField("", []);
    expect(findButton(el, "Add stage")).toBeNull();
    expect(el.textContent ?? "").toContain("Add resolutions above first");
    await unmount();
  });

  it("renders one row per stage with normalized percentages", async () => {
    const initial = JSON.stringify({
      enabled: true,
      stage: [
        { resolutions: [512], fraction: 1 },
        { resolutions: [768], fraction: 1 },
        { resolutions: [1024], fraction: 1 },
      ],
    });
    const { el, unmount } = await mountField(initial, [512, 768, 1024]);
    expect(el.querySelectorAll(".stage-row")).toHaveLength(3);
    const percents = Array.from(el.querySelectorAll(".stage-percent")).map(
      (n) => n.textContent?.trim() ?? ""
    );
    expect(percents).toEqual(["33%", "33%", "33%"]);
    await unmount();
  });

  it("warns when a stage uses a resolution not in the dataset list", async () => {
    const initial = JSON.stringify({
      enabled: true,
      stage: [{ resolutions: [333], fraction: 1 }],
    });
    const { el, unmount } = await mountField(initial, [512, 768]);
    expect(el.textContent ?? "").toContain("not in the dataset's Resolutions list");
    await unmount();
  });
});

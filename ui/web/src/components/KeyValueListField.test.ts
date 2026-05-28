import { describe, expect, it } from "vitest";
import { createApp, nextTick, ref } from "vue";
import ElementPlus from "element-plus";
import KeyValueListField from "./KeyValueListField.vue";

describe("KeyValueListField", () => {
  async function mountField(initial: unknown = "") {
    const model = ref(initial);
    const el = document.createElement("div");
    document.body.appendChild(el);

    const app = createApp({
      components: { KeyValueListField },
      setup() {
        return { model };
      },
      template: '<KeyValueListField v-model="model" />',
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

  it("keeps a row visible while typing an incomplete parameter", async () => {
    const { el, model, unmount } = await mountField("");
    const keyInput = el.querySelector(".kv-key input") as HTMLInputElement;
    expect(keyInput).toBeTruthy();

    keyInput.value = "warmup_steps";
    keyInput.dispatchEvent(new InputEvent("input", { bubbles: true }));
    await nextTick();
    await nextTick();

    expect(el.querySelectorAll(".kv-row")).toHaveLength(1);
    expect(keyInput.value).toBe("warmup_steps");
    expect(model.value).toBe("");

    await unmount();
  });

  it("does not render inline runtime token glossary", async () => {
    const el = document.createElement("div");
    document.body.appendChild(el);

    const app = createApp({
      components: { KeyValueListField },
      template: `
        <KeyValueListField
          :runtime-tokens="['total_steps', 'effective_total_steps']"
        />
      `,
    });
    app.use(ElementPlus);
    app.mount(el);
    await nextTick();

    expect(el.querySelector(".kv-runtime-tokens")).toBeNull();
    expect(el.textContent ?? "").not.toContain("min(total_steps");

    app.unmount();
    el.remove();
  });

  it("offers runtime tokens in value autocomplete", async () => {
    const el = document.createElement("div");
    document.body.appendChild(el);

    const app = createApp({
      components: { KeyValueListField },
      template: `
        <KeyValueListField :runtime-tokens="['total_steps', 'epochs']" />
      `,
    });
    app.use(ElementPlus);
    app.mount(el);
    await nextTick();

    expect(el.querySelector(".kv-value")).toBeTruthy();
    expect(el.querySelector(".kv-value input")?.getAttribute("placeholder")).toBe(
      "Value or runtime token"
    );

    app.unmount();
    el.remove();
  });

  it("emits completed key-value pairs", async () => {
    const { el, model, unmount } = await mountField("");
    const keyInput = el.querySelector(".kv-key input") as HTMLInputElement;
    const valueInput = el.querySelector(".kv-value input") as HTMLInputElement;

    keyInput.value = "lr_min";
    keyInput.dispatchEvent(new InputEvent("input", { bubbles: true }));
    valueInput.value = "0.01";
    valueInput.dispatchEvent(new InputEvent("input", { bubbles: true }));
    await nextTick();
    await nextTick();

    expect(model.value).toEqual({ lr_min: 0.01 });

    await unmount();
  });
});

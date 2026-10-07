import { afterEach, describe, expect, it } from "vitest";
import { createApp, h, nextTick, ref } from "vue";
import ElementPlus from "element-plus";
import CaptionOutputFormat, { type CaptionOutputValue } from "./CaptionOutputFormat.vue";

async function mountWith(value: CaptionOutputValue) {
  const model = ref(value);
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({
    render: () =>
      h(CaptionOutputFormat, {
        modelValue: model.value,
        "onUpdate:modelValue": (v: CaptionOutputValue) => (model.value = v),
      }),
  });
  app.use(ElementPlus);
  app.mount(el);
  for (let i = 0; i < 4; i += 1) await nextTick();
  return { app, el };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("CaptionOutputFormat extension warning", () => {
  it("warns that training reads only .txt sidecars when another extension is chosen", async () => {
    const { app, el } = await mountWith({ format: "sidecar", ext: ".caption" });
    expect(el.textContent).toContain("Training reads only");
    app.unmount();
  });

  it.each([
    [{ format: "sidecar", ext: ".txt" }],
    [{ format: "sidecar", ext: "TXT" }],
    [{ format: "json", ext: ".caption" }],
    [{ format: "inherit", ext: ".caption" }],
  ] as const)("stays quiet for %j", async (value) => {
    const { app, el } = await mountWith({ ...value });
    expect(el.textContent).not.toContain("Training reads only");
    app.unmount();
  });
});

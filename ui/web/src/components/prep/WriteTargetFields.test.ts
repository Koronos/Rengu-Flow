/** The target line + write mode control shared by the tag, caption and edit-instruction forms. */
import { afterEach, describe, expect, it } from "vitest";
import { createApp, h, nextTick, reactive } from "vue";
import ElementPlus, { ElForm } from "element-plus";
import WriteTargetFields from "./WriteTargetFields.vue";
import { buildStageConfig, defaultCommonForm, defaultTagForm } from "../../lib/prepStageConfig";

async function mountFields(form: ReturnType<typeof defaultTagForm>, stage: "tag" | "caption" = "tag") {
  const el = document.createElement("div");
  document.body.appendChild(el);
  const app = createApp({
    render: () => h(ElForm, null, { default: () => h(WriteTargetFields, { form, stage }) }),
  });
  app.use(ElementPlus);
  app.mount(el);
  for (let i = 0; i < 6; i += 1) await nextTick();
  return { app, el };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("WriteTargetFields", () => {
  it("shows the mode the legacy overwrite flag implies, without writing anything", async () => {
    const form = reactive({ ...defaultTagForm(), overwrite: true });
    const { app, el } = await mountFields(form);

    expect(el.textContent).toContain("Replace it");
    expect(form.write_mode).toBe(""); // merely shown: the config stays as it was saved
    expect(buildStageConfig("tag", { form: defaultCommonForm(), tagForm: form }).tag).not.toHaveProperty(
      "write_mode",
    );

    app.unmount();
  });

  it("accepts any line number, with no upper bound", async () => {
    const form = reactive({ ...defaultTagForm(), target_line: 42 });
    const { app } = await mountFields(form);

    const input = document.querySelector<HTMLInputElement>(".el-input-number input");
    expect(input?.value).toBe("42");
    expect(document.querySelector(".el-input-number")?.className).not.toContain("is-disabled");

    app.unmount();
  });

  it("choosing a mode stores it and keeps overwrite in step", async () => {
    const form = reactive(defaultTagForm());
    const { app } = await mountFields(form);

    document.querySelector<HTMLElement>(".el-select__wrapper")?.click();
    for (let i = 0; i < 6; i += 1) await nextTick();
    const options = [...document.querySelectorAll<HTMLElement>(".el-select-dropdown__item")];
    options.find((o) => (o.textContent ?? "").includes("Append"))?.click();
    for (let i = 0; i < 4; i += 1) await nextTick();

    expect(form.write_mode).toBe("append");
    expect(form.overwrite).toBe(false);
    const built = buildStageConfig("tag", { form: defaultCommonForm(), tagForm: form }).tag;
    expect(built).toMatchObject({ write_mode: "append", overwrite: false });

    app.unmount();
  });
});

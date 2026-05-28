import { describe, expect, it, vi } from "vitest";
import { createApp, nextTick } from "vue";
import { createPinia, setActivePinia } from "pinia";
import ElementPlus from "element-plus";
import DatasetFormFolders from "./DatasetFormFolders.vue";
import { useDatasetEditorStore } from "../stores/datasetEditor";

describe("DatasetFormFolders", () => {
  it("mounts and lists directories from editor form", async () => {
    const pinia = createPinia();
    setActivePinia(pinia);

    const editor = useDatasetEditorStore();
    editor.form = {
      _directories: [{ path: "/data/sample", num_repeats: 1 }],
      resolutions: [1024],
      frame_buckets: [1],
    };
    editor.schema = { sections: [], directory_fields: [] };
    editor.content =
      'resolutions = [1024]\nframe_buckets = [1]\n\n[[directory]]\npath = "/data/sample"\nnum_repeats = 1\n';

    const errors: unknown[] = [];
    const el = document.createElement("div");
    document.body.appendChild(el);

    const app = createApp(DatasetFormFolders);
    app.use(pinia);
    app.use(ElementPlus);
    app.config.errorHandler = (err) => {
      errors.push(err);
    };

    app.mount(el);
    await nextTick();
    await nextTick();

    expect(errors).toEqual([]);
    expect(el.querySelector(".dataset-folders")).toBeTruthy();
    expect(el.textContent).toContain("1 directory");
    expect(el.textContent).toContain("sample");

    app.unmount();
    el.remove();
  });
});

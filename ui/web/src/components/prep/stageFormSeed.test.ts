/**
 * The one rule the two registry-fetching stage forms share: **a seeded form is never written
 * over.**
 *
 * Both own an `api.prepModels` call whose result preselects something into the *shared* v-model
 * object. That object belongs to the parent — in the workflow drawer the parent watches it,
 * emits `update:node` and autosaves 700 ms later — so a preselect that lands on top of a saved
 * value is a lost update, and workflows keep no history to recover it from. Merely opening a
 * saved step must therefore leave every seeded field untouched, not even for one tick.
 *
 * The watchers below are `flush: "sync"`, which is the point: a batched watcher only reports
 * where the value *settled*, and this bug is about where it passed through. A transient
 * preselect is exactly what reaches the parent.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp, nextTick, reactive, watch } from "vue";
import ElementPlus from "element-plus";
import TagStageForm from "./TagStageForm.vue";
import CaptionStageForm from "./CaptionStageForm.vue";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCommonForm,
  defaultTagForm,
  type ModelThresholds,
  type PrepCaptionForm,
  type PrepTagForm,
} from "../../lib/prepStageConfig";
import { api } from "../../api";
import type { PrepCaptionConfig, PrepTagConfig } from "../../types/api";

vi.mock("../../api", () => ({
  api: {
    prepModels: vi.fn(async () => ({ models: [] })),
    prepCaptionPrompts: vi.fn(async () => promptCatalogue()),
    prepCaptionPromptPreview: vi.fn(async () => ({ prompt: "", native_format: false })),
  },
}));

function promptCatalogue() {
  return {
    bases: [],
    modifiers: [],
    outfit_modes: ["describe", "omit", "mixed"],
    default_base: "registry-base",
    default_modifiers: ["registry-modifier"],
    no_meta: "",
    character_trigger_template: "",
    outfit_texts: {},
    sampling_defaults: {},
  };
}

/** A registry whose preselect ("downloaded-A") is deliberately not what the seeds carry. */
const TAG_REGISTRY = [
  { id: "downloaded-A", repo_id: "r/a", downloaded: true, available: true },
  { id: "saved-B", repo_id: "r/b", downloaded: false, available: true },
];

const CAPTION_REGISTRY = [
  { id: "registry-first", repo_id: "r/first", downloaded: true, available: true },
  { id: "saved-captioner", repo_id: "r/saved", downloaded: true, available: true },
];

function savedTagSeed(): PrepTagConfig {
  const payload = buildStageConfig("tag", {
    form: defaultCommonForm(),
    tagForm: { ...defaultTagForm(), models: ["saved-B"], max_tags: 55 },
    tagThresholds: { "saved-B": { general: 0.4, character: 0.9, rating: 0.6 } },
  });
  return payload.tag as PrepTagConfig;
}

function savedCaptionSeed(): PrepCaptionConfig {
  const payload = buildStageConfig("caption", {
    form: defaultCommonForm(),
    captionForm: {
      ...defaultCaptionForm(),
      model: "saved-captioner",
      prompt_base: "saved-base",
      prompt_modifiers: ["saved-modifier"],
      character_name: "Aoi",
    },
  });
  return payload.caption as PrepCaptionConfig;
}

/** Mount a form the way its parents do: one shared, reactive form object plus a seed. */
function mount(component: unknown, props: Record<string, unknown>) {
  const el = document.createElement("div");
  document.body.appendChild(el);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const app = createApp(component as any, props);
  app.use(ElementPlus);
  app.mount(el);
  return app;
}

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i += 1) await nextTick();
}

beforeEach(() => {
  vi.mocked(api.prepModels).mockResolvedValue({ models: [] });
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("TagStageForm seeding", () => {
  it("never writes the registry preselect over a seeded model list", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: TAG_REGISTRY });
    const form = reactive<PrepTagForm>(defaultTagForm());
    const thresholds = reactive<Record<string, ModelThresholds>>({});
    const writes: string[][] = [];
    watch(() => form.models, (models) => writes.push([...models]), { flush: "sync" });

    const app = mount(TagStageForm, { modelValue: form, thresholds, seed: savedTagSeed() });
    await settle();

    // The seed is the only thing that ever assigned `models`; `downloaded-A` never appears.
    expect(writes).toEqual([["saved-B"]]);
    expect(form.max_tags).toBe(55);
    expect(thresholds["saved-B"]).toEqual({ general: 0.4, character: 0.9, rating: 0.6 });

    app.unmount();
  });

  it("still preselects the downloaded models when there is nothing to protect", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: TAG_REGISTRY });
    const form = reactive<PrepTagForm>(defaultTagForm());
    const thresholds = reactive<Record<string, ModelThresholds>>({});

    const app = mount(TagStageForm, { modelValue: form, thresholds, seed: null });
    await settle();

    expect(form.models).toEqual(["downloaded-A"]);

    app.unmount();
  });

  it("fills the gap when the seed itself selected nothing", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: TAG_REGISTRY });
    const form = reactive<PrepTagForm>(defaultTagForm());
    const thresholds = reactive<Record<string, ModelThresholds>>({});
    const seed = { ...savedTagSeed(), models: [], overrides: {} };

    const app = mount(TagStageForm, { modelValue: form, thresholds, seed });
    await settle();

    // A step saved with no tagger runs with no tagger, so offering the downloaded ones here is
    // filling a gap, not overriding a choice — and the user sees it before it is saved.
    expect(form.models).toEqual(["downloaded-A"]);
    expect(form.max_tags).toBe(55);

    app.unmount();
  });
});

describe("CaptionStageForm seeding", () => {
  it("never writes the registry's model or prompt defaults over a seeded form", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: CAPTION_REGISTRY });
    const form = reactive<PrepCaptionForm>(defaultCaptionForm());
    const models: string[] = [];
    const bases: string[] = [];
    const modifiers: string[][] = [];
    watch(() => form.model, (v) => models.push(v), { flush: "sync" });
    watch(() => form.prompt_base, (v) => bases.push(v), { flush: "sync" });
    watch(() => form.prompt_modifiers, (v) => modifiers.push([...v]), { flush: "sync" });

    const app = mount(CaptionStageForm, { modelValue: form, seed: savedCaptionSeed() });
    await settle();

    expect(models).toEqual(["saved-captioner"]);
    expect(bases).toEqual(["saved-base"]);
    expect(modifiers).toEqual([["saved-modifier"]]);
    expect(form.character_name).toBe("Aoi");

    app.unmount();
  });

  it("takes the registry's first model and prompt defaults for an unseeded form", async () => {
    vi.mocked(api.prepModels).mockResolvedValue({ models: CAPTION_REGISTRY });
    const form = reactive<PrepCaptionForm>(defaultCaptionForm());

    const app = mount(CaptionStageForm, { modelValue: form, seed: null });
    await settle();

    expect(form.model).toBe("registry-first");
    expect(form.prompt_base).toBe("registry-base");
    expect(form.prompt_modifiers).toEqual(["registry-modifier"]);

    app.unmount();
  });
});

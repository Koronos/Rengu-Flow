import { describe, expect, it } from "vitest";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCleanForm,
  defaultCommonForm,
  defaultQualityForm,
  defaultTagForm,
  modelThresholdDefaults,
  type PrepStageForms,
} from "./prepStageConfig";
import type { PrepModelInfo } from "../types/api";

/**
 * The expectations below are GOLDEN: they were captured by running the original
 * `buildConfig()` out of `PrepJobFormView.vue` before it was extracted. Any diff
 * here is a behaviour change in the prep job payload, not a test that needs
 * updating. The single deliberate exception is the `index` stage, which had no
 * branch at all and used to fall through to `clean`.
 */

const TAG_MODELS: PrepModelInfo[] = [
  {
    id: "wd-eva02-large-tagger-v3",
    repo_id: "SmilingWolf/wd-eva02-large-tagger-v3",
    downloaded: true,
    available: true,
    general_threshold: 0.35,
    character_threshold: 0.85,
    rating_threshold: 0.5,
  },
  {
    id: "wd-swinv2-tagger-v3",
    repo_id: "SmilingWolf/wd-swinv2-tagger-v3",
    downloaded: false,
    available: true,
    general_threshold: 0.4,
    character_threshold: 0.8,
    rating_threshold: 0.45,
  },
];

const commonForm = () => ({ ...defaultCommonForm(), path: "/data/ds" });

function forms(overrides: Partial<PrepStageForms> = {}): PrepStageForms {
  return { form: commonForm(), tagModels: TAG_MODELS, ...overrides };
}

describe("buildStageConfig — common fields", () => {
  it("carries path and caption layout onto every stage", () => {
    const cfg = buildStageConfig("clean", forms());
    expect(cfg.path).toBe("/data/ds");
    expect(cfg.caption_format).toBe("sidecar");
    expect(cfg.caption_ext).toBe(".txt");
  });

  it("falls back to .txt when the caption extension is blank", () => {
    const cfg = buildStageConfig("clean", forms({ form: { ...commonForm(), caption_ext: "" } }));
    expect(cfg.caption_ext).toBe(".txt");
  });

  it("keeps a non-default caption extension and format", () => {
    const cfg = buildStageConfig(
      "caption",
      forms({ form: { path: "/x", caption_format: "json", caption_ext: ".caption" } })
    );
    expect(cfg.caption_format).toBe("json");
    expect(cfg.caption_ext).toBe(".caption");
  });

  it("emits exactly one stage section", () => {
    for (const stage of ["tag", "caption", "clean", "quality", "index"] as const) {
      const cfg = buildStageConfig(stage, forms());
      const sections = ["tag", "caption", "clean", "quality", "index"].filter(
        (k) => (cfg as unknown as Record<string, unknown>)[k] !== undefined
      );
      expect(sections).toEqual([stage]);
    }
  });
});

describe("buildStageConfig — tag (golden)", () => {
  it("writes one override block per model, seeded from its thresholds", () => {
    const cfg = buildStageConfig(
      "tag",
      forms({
        tagForm: { ...defaultTagForm(), models: ["wd-eva02-large-tagger-v3", "wd-swinv2-tagger-v3"] },
        tagThresholds: {
          "wd-eva02-large-tagger-v3": { general: 0.35, character: 0.85, rating: 0.5 },
          "wd-swinv2-tagger-v3": { general: 0.4, character: 0.8, rating: 0.45 },
        },
      })
    );
    expect(cfg).toEqual({
      path: "/data/ds",
      caption_format: "sidecar",
      caption_ext: ".txt",
      tag: {
        models: ["wd-eva02-large-tagger-v3", "wd-swinv2-tagger-v3"],
        exclude_tags: [],
        prepend_tags: [],
        max_tags: 40,
        batch_size: 8,
        overwrite: false,
        quality_tags: false,
        underscores: false,
        target_line: 1,
        overrides: {
          "wd-eva02-large-tagger-v3": {
            general_threshold: 0.35,
            character_threshold: 0.85,
            rating_threshold: 0.5,
          },
          "wd-swinv2-tagger-v3": {
            general_threshold: 0.4,
            character_threshold: 0.8,
            rating_threshold: 0.45,
          },
        },
      },
    });
  });

  it("turns a 0 floor into include_character/include_rating = false", () => {
    const cfg = buildStageConfig(
      "tag",
      forms({
        form: { path: "/data/ds", caption_format: "json", caption_ext: "" },
        tagForm: {
          ...defaultTagForm(),
          models: ["wd-eva02-large-tagger-v3"],
          exclude_tags: ["realistic", "3d"],
          prepend_tags: ["trigger"],
          max_tags: 25,
          batch_size: 16,
          overwrite: true,
          quality_tags: true,
          underscores: true,
          target_line: 3,
        },
        tagThresholds: { "wd-eva02-large-tagger-v3": { general: 0.2, character: 0, rating: 0 } },
      })
    );
    expect(cfg).toEqual({
      path: "/data/ds",
      caption_format: "json",
      caption_ext: ".txt",
      tag: {
        models: ["wd-eva02-large-tagger-v3"],
        exclude_tags: ["realistic", "3d"],
        prepend_tags: ["trigger"],
        max_tags: 25,
        batch_size: 16,
        overwrite: true,
        quality_tags: true,
        underscores: true,
        target_line: 3,
        overrides: {
          "wd-eva02-large-tagger-v3": {
            general_threshold: 0.2,
            include_character: false,
            include_rating: false,
          },
        },
      },
    });
  });

  it("falls back to the model's registry defaults when a thresholds row is missing", () => {
    const cfg = buildStageConfig(
      "tag",
      forms({
        tagForm: { ...defaultTagForm(), models: ["wd-swinv2-tagger-v3", "unknown-model"] },
        tagThresholds: {},
      })
    );
    expect(cfg.tag?.overrides).toEqual({
      "wd-swinv2-tagger-v3": {
        general_threshold: 0.4,
        character_threshold: 0.8,
        rating_threshold: 0.45,
      },
      // unknown model -> registry-wide defaults
      "unknown-model": {
        general_threshold: 0.35,
        character_threshold: 0.85,
        rating_threshold: 0.5,
      },
    });
  });

  it("emits an empty override map when no model is selected", () => {
    const cfg = buildStageConfig("tag", forms({ tagForm: defaultTagForm(), tagThresholds: {} }));
    expect(cfg.tag?.models).toEqual([]);
    expect(cfg.tag?.overrides).toEqual({});
  });

  it("copies the tag arrays instead of aliasing the form state", () => {
    const tagForm = { ...defaultTagForm(), models: ["a"], exclude_tags: ["x"], prepend_tags: ["y"] };
    const cfg = buildStageConfig("tag", forms({ tagForm, tagThresholds: {} }));
    expect(cfg.tag?.models).not.toBe(tagForm.models);
    expect(cfg.tag?.exclude_tags).not.toBe(tagForm.exclude_tags);
    expect(cfg.tag?.prepend_tags).not.toBe(tagForm.prepend_tags);
  });
});

describe("buildStageConfig — caption (golden)", () => {
  it("passes the whole caption form through unchanged", () => {
    const cfg = buildStageConfig(
      "caption",
      forms({ captionForm: { ...defaultCaptionForm(), model: "joycaption-beta-one" } })
    );
    expect(cfg.caption).toEqual({
      model: "joycaption-beta-one",
      quantization: "bf16",
      prompt: "",
      prompt_base: "descriptive-long",
      prompt_modifiers: ["demographics"],
      character_name: "",
      character_canon: "",
      outfit: "describe",
      target_line: 2,
      max_new_tokens: 512,
      temperature: null,
      top_p: null,
      exact_generation: false,
      batch_size: 4,
      use_tags_as_grounding: true,
      overwrite: false,
      max_image_side: 1536,
      min_image_side: 0,
      engine: "hf",
      vllm_quantization: "gptq",
      vllm_model: "",
      gguf_quantization: "Q8_0",
    });
  });

  it("keeps every vllm / character / sampling field", () => {
    const cfg = buildStageConfig(
      "caption",
      forms({
        form: { path: "/x", caption_format: "sidecar", caption_ext: ".caption" },
        captionForm: {
          ...defaultCaptionForm(),
          model: "joycaption-beta-one",
          engine: "vllm",
          vllm_quantization: "awq",
          vllm_model: "some/repo",
          prompt: "my own prompt",
          prompt_modifiers: ["demographics", "lighting"],
          character_name: "hatsune miku",
          character_canon: "aqua twin-tails",
          outfit: "mixed",
          temperature: 0.6,
          top_p: 0.9,
          exact_generation: true,
          overwrite: true,
          target_line: 3,
          min_image_side: 256,
        },
      })
    );
    expect(cfg).toEqual({
      path: "/x",
      caption_format: "sidecar",
      caption_ext: ".caption",
      caption: {
        model: "joycaption-beta-one",
        quantization: "bf16",
        prompt: "my own prompt",
        prompt_base: "descriptive-long",
        prompt_modifiers: ["demographics", "lighting"],
        character_name: "hatsune miku",
        character_canon: "aqua twin-tails",
        outfit: "mixed",
        target_line: 3,
        max_new_tokens: 512,
        temperature: 0.6,
        top_p: 0.9,
        exact_generation: true,
        batch_size: 4,
        use_tags_as_grounding: true,
        overwrite: true,
        max_image_side: 1536,
        min_image_side: 256,
        engine: "vllm",
        vllm_quantization: "awq",
        vllm_model: "some/repo",
        gguf_quantization: "Q8_0",
      },
    });
  });

  it("keeps the gguf fields for ToriiGate", () => {
    const cfg = buildStageConfig(
      "caption",
      forms({
        captionForm: {
          ...defaultCaptionForm(),
          model: "toriigate-0.5",
          engine: "gguf",
          gguf_quantization: "Q5_K_M",
          use_tags_as_grounding: false,
        },
      })
    );
    expect(cfg.caption?.engine).toBe("gguf");
    expect(cfg.caption?.gguf_quantization).toBe("Q5_K_M");
    expect(cfg.caption?.use_tags_as_grounding).toBe(false);
  });

  it("copies prompt_modifiers instead of aliasing the form state", () => {
    const captionForm = { ...defaultCaptionForm(), prompt_modifiers: ["demographics"] };
    const cfg = buildStageConfig("caption", forms({ captionForm }));
    expect(cfg.caption?.prompt_modifiers).not.toBe(captionForm.prompt_modifiers);
  });
});

describe("buildStageConfig — clean (golden)", () => {
  it("passes the clean form through unchanged", () => {
    const cfg = buildStageConfig("clean", forms({ cleanForm: defaultCleanForm() }));
    expect(cfg.clean).toEqual({
      confidence: 0.35,
      mask_dilation_px: 8,
      in_place: false,
      output_dir: "",
      copy_undetected: true,
    });
  });

  it("keeps in-place and the output dir", () => {
    const cfg = buildStageConfig(
      "clean",
      forms({
        cleanForm: {
          confidence: 0.6,
          mask_dilation_px: 20,
          in_place: true,
          output_dir: "/out",
          copy_undetected: false,
        },
      })
    );
    expect(cfg.clean).toEqual({
      confidence: 0.6,
      mask_dilation_px: 20,
      in_place: true,
      output_dir: "/out",
      copy_undetected: false,
    });
  });
});

describe("buildStageConfig — quality (golden)", () => {
  it("maps move=false to action 'report'", () => {
    const cfg = buildStageConfig("quality", forms({ qualityForm: defaultQualityForm() }));
    expect(cfg.quality).toEqual({
      metric: "blur",
      blur_threshold: 80,
      min_side: 0,
      min_detail: 0,
      aesthetic_min_label: "normal",
      iqa_model: "clipiqa",
      iqa_threshold: 10,
      action: "report",
      output_dir: "",
    });
  });

  it("maps move=true to action 'move' and keeps every metric field", () => {
    const cfg = buildStageConfig(
      "quality",
      forms({
        qualityForm: {
          ...defaultQualityForm(),
          metric: "iqa",
          iqa_model: "arniqa",
          iqa_threshold: 35,
          move: true,
          output_dir: "/rejects",
          min_detail: 12.5,
          min_side: 512,
          blur_threshold: 95,
          aesthetic_min_label: "good",
        },
      })
    );
    expect(cfg.quality).toEqual({
      metric: "iqa",
      blur_threshold: 95,
      min_side: 512,
      min_detail: 12.5,
      aesthetic_min_label: "good",
      iqa_model: "arniqa",
      iqa_threshold: 35,
      action: "move",
      output_dir: "/rejects",
    });
  });

  it("never emits the form-only `move` key", () => {
    const cfg = buildStageConfig("quality", forms({ qualityForm: defaultQualityForm() }));
    expect(cfg.quality).not.toHaveProperty("move");
  });
});

describe("buildStageConfig — index", () => {
  // NOT a golden: `index` had no branch and fell through to `clean`, which the
  // server rejects ("index stage needs at least one model in [index].models").
  it("emits the index models instead of falling through to clean", () => {
    const cfg = buildStageConfig("index", forms({ indexForm: { models: ["aesthetic", "clipiqa"] } }));
    expect(cfg).toEqual({
      path: "/data/ds",
      caption_format: "sidecar",
      caption_ext: ".txt",
      index: { models: ["aesthetic", "clipiqa"] },
    });
    expect(cfg.clean).toBeUndefined();
  });

  it("copies the model list instead of aliasing the form state", () => {
    const indexForm = { models: ["aesthetic"] };
    const cfg = buildStageConfig("index", forms({ indexForm }));
    expect(cfg.index?.models).not.toBe(indexForm.models);
  });
});

describe("modelThresholdDefaults", () => {
  it("uses the model's own floors", () => {
    expect(modelThresholdDefaults(TAG_MODELS[1])).toEqual({
      general: 0.4,
      character: 0.8,
      rating: 0.45,
    });
  });

  it("falls back to the registry-wide floors for an unknown model", () => {
    expect(modelThresholdDefaults(undefined)).toEqual({
      general: 0.35,
      character: 0.85,
      rating: 0.5,
    });
  });
});

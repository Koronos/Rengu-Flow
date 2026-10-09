/**
 * Prep stage form state and the pure config builder.
 *
 * Extracted verbatim from `PrepJobFormView.vue`'s `buildConfig()` so the same
 * payload can be produced outside a view (workflow node drawer). The shapes
 * here are the *form* shapes — notably `quality.move` (a switch) which the
 * builder maps to the server's `quality.action`.
 */
import type { PrepConfigDto, PrepModelInfo, PrepStage } from "../types/api";

/** Per-model confidence floors; 0 means "drop that category for this model". */
export interface ModelThresholds {
  general: number;
  character: number;
  rating: number;
}

/**
 * What a step does with a target line that already has text. `""` = never chosen: the server then
 * derives it from the legacy `overwrite` flag (on = replace, off = skip), and the config carries
 * no `write_mode` key at all - so a step that was only opened is saved exactly as it was.
 */
export type WriteMode = "" | "skip" | "replace" | "append";

export interface PrepCommonForm {
  path: string;
  caption_format: "sidecar" | "json";
  caption_ext: string;
}

export interface PrepTagForm {
  models: string[];
  exclude_tags: string[];
  prepend_tags: string[];
  max_tags: number;
  batch_size: number;
  overwrite: boolean;
  write_mode: WriteMode;
  quality_tags: boolean;
  underscores: boolean;
  target_line: number;
}

export interface PrepCaptionForm {
  model: string;
  quantization: "bf16" | "int8" | "nf4";
  prompt: string;
  prompt_base: string;
  prompt_modifiers: string[];
  character_name: string;
  character_canon: string;
  outfit: "describe" | "omit" | "mixed";
  target_line: number;
  max_new_tokens: number;
  temperature: number | null;
  top_p: number | null;
  exact_generation: boolean;
  batch_size: number;
  use_tags_as_grounding: boolean;
  /** 1-based caption line the grounding tags are read from (default 1, the tag line). */
  tags_line: number;
  overwrite: boolean;
  write_mode: WriteMode;
  max_image_side: number;
  min_image_side: number;
  engine: "hf" | "vllm" | "gguf";
  vllm_quantization: "gptq" | "fp8" | "awq" | "none";
  vllm_model: string;
  gguf_quantization: "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M";
}

export interface PrepEditCaptionForm {
  control_path: string;
  model: string;
  gguf_quantization: "" | "Q8_0" | "Q6_K" | "Q5_K_M" | "Q4_K_M";
  prompt: string;
  overwrite: boolean;
  write_mode: WriteMode;
  /** 1-based caption line the instruction is written to (default 1). */
  target_line: number;
  max_pixels: number;
  max_new_tokens: number;
  temperature: number | null;
  top_p: number | null;
  n_parallel: number;
}

export interface PrepCleanForm {
  confidence: number;
  mask_dilation_px: number;
  in_place: boolean;
  output_dir: string;
  copy_undetected: boolean;
}

export interface PrepQualityForm {
  metric: "blur" | "aesthetic" | "iqa";
  blur_threshold: number;
  min_side: number;
  min_detail: number;
  aesthetic_min_label: string;
  iqa_model: string;
  iqa_threshold: number;
  /** Switch state; maps to `action: "move" | "report"`. */
  move: boolean;
  output_dir: string;
}

export interface PrepIndexForm {
  models: string[];
}

/** Everything `buildStageConfig` may read; only the stage in play has to be set. */
export interface PrepStageForms {
  form: PrepCommonForm;
  tagForm?: PrepTagForm;
  tagThresholds?: Record<string, ModelThresholds>;
  /** Registry entries, used only to seed a missing thresholds row. */
  tagModels?: PrepModelInfo[];
  captionForm?: PrepCaptionForm;
  editCaptionForm?: PrepEditCaptionForm;
  cleanForm?: PrepCleanForm;
  qualityForm?: PrepQualityForm;
  indexForm?: PrepIndexForm;
}

export const defaultCommonForm = (): PrepCommonForm => ({
  path: "",
  caption_format: "sidecar",
  caption_ext: ".txt",
});

export const defaultTagForm = (): PrepTagForm => ({
  models: [],
  exclude_tags: [],
  prepend_tags: [],
  max_tags: 40,
  batch_size: 8,
  overwrite: false,
  write_mode: "",
  quality_tags: false,
  underscores: false,
  target_line: 1,
});

export const defaultCaptionForm = (): PrepCaptionForm => ({
  model: "",
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
  tags_line: 1,
  overwrite: false,
  write_mode: "",
  max_image_side: 1536,
  min_image_side: 0,
  engine: "hf",
  vllm_quantization: "gptq",
  vllm_model: "",
  gguf_quantization: "Q8_0",
});

export const defaultEditCaptionForm = (): PrepEditCaptionForm => ({
  control_path: "",
  model: "",
  gguf_quantization: "",
  prompt: "",
  overwrite: false,
  write_mode: "",
  target_line: 1,
  max_pixels: 524288,
  max_new_tokens: 96,
  temperature: 0.2,
  top_p: null,
  n_parallel: 0,
});

export const defaultCleanForm = (): PrepCleanForm => ({
  confidence: 0.35,
  mask_dilation_px: 8,
  in_place: false,
  output_dir: "",
  copy_undetected: true,
});

export const defaultQualityForm = (): PrepQualityForm => ({
  metric: "blur",
  blur_threshold: 80,
  min_side: 0,
  min_detail: 0,
  aesthetic_min_label: "normal",
  iqa_model: "clipiqa",
  iqa_threshold: 10,
  move: false,
  output_dir: "",
});

export const defaultIndexForm = (): PrepIndexForm => ({ models: [] });

/**
 * A model's own confidence floors, falling back to the registry-wide defaults
 * when the model is unknown (or the registry has not loaded yet).
 */
export function modelThresholdDefaults(model: PrepModelInfo | undefined): ModelThresholds {
  return {
    general: model?.general_threshold ?? 0.35,
    character: model?.character_threshold ?? 0.85,
    rating: model?.rating_threshold ?? 0.5,
  };
}

/** The legacy `overwrite` flag a form implies: an explicit mode wins, replace is overwrite. */
export function effectiveOverwrite(f: { write_mode: WriteMode; overwrite: boolean }): boolean {
  return f.write_mode ? f.write_mode === "replace" : f.overwrite;
}

/** `{ write_mode }` once a mode was chosen, nothing before - see {@link WriteMode}. */
export function writeModeKey(f: { write_mode: WriteMode }): { write_mode?: "skip" | "replace" | "append" } {
  return f.write_mode ? { write_mode: f.write_mode } : {};
}

/** `{ tags_line }` once it leaves the default (line 1) - the same rule as the other late fields. */
export function tagsLineKey(f: { tags_line: number }): { tags_line?: number } {
  return f.tags_line !== 1 ? { tags_line: f.tags_line } : {};
}

/**
 * Build the `config` payload for a prep job from the form state.
 *
 * Pure: same inputs, same dict. `caption_ext` falls back to `.txt` when blank.
 */
export function buildStageConfig(stage: PrepStage, forms: PrepStageForms): PrepConfigDto {
  const form = forms.form;
  const base = {
    path: form.path,
    caption_format: form.caption_format,
    caption_ext: form.caption_ext || ".txt",
  };

  if (stage === "tag") {
    const tagForm = forms.tagForm ?? defaultTagForm();
    const tagThresholds = forms.tagThresholds ?? {};
    // Each model carries its own confidence floors as a per-model override; a 0 floor
    // for character/rating turns that category off for that model (general is kept).
    const overrides: Record<string, Record<string, number | boolean>> = {};
    for (const id of tagForm.models) {
      const t =
        tagThresholds[id] ??
        modelThresholdDefaults((forms.tagModels ?? []).find((m) => m.id === id));
      const o: Record<string, number | boolean> = { general_threshold: t.general };
      if (t.character > 0) o.character_threshold = t.character;
      else o.include_character = false;
      if (t.rating > 0) o.rating_threshold = t.rating;
      else o.include_rating = false;
      overrides[id] = o;
    }
    return {
      ...base,
      tag: {
        models: [...tagForm.models],
        exclude_tags: [...tagForm.exclude_tags],
        prepend_tags: [...tagForm.prepend_tags],
        max_tags: tagForm.max_tags,
        batch_size: tagForm.batch_size,
        overwrite: effectiveOverwrite(tagForm),
        ...writeModeKey(tagForm),
        quality_tags: tagForm.quality_tags,
        underscores: tagForm.underscores,
        target_line: tagForm.target_line,
        overrides,
      },
    };
  }
  if (stage === "caption") {
    const captionForm = forms.captionForm ?? defaultCaptionForm();
    return {
      ...base,
      caption: {
        model: captionForm.model,
        quantization: captionForm.quantization,
        prompt: captionForm.prompt,
        prompt_base: captionForm.prompt_base,
        prompt_modifiers: [...captionForm.prompt_modifiers],
        character_name: captionForm.character_name,
        character_canon: captionForm.character_canon,
        outfit: captionForm.outfit,
        target_line: captionForm.target_line,
        max_new_tokens: captionForm.max_new_tokens,
        temperature: captionForm.temperature,
        top_p: captionForm.top_p,
        exact_generation: captionForm.exact_generation,
        batch_size: captionForm.batch_size,
        use_tags_as_grounding: captionForm.use_tags_as_grounding,
        // Omitted at its default so a caption step that was only opened keeps its saved config.
        ...tagsLineKey(captionForm),
        overwrite: effectiveOverwrite(captionForm),
        ...writeModeKey(captionForm),
        max_image_side: captionForm.max_image_side,
        min_image_side: captionForm.min_image_side,
        engine: captionForm.engine,
        vllm_quantization: captionForm.vllm_quantization,
        vllm_model: captionForm.vllm_model,
        gguf_quantization: captionForm.gguf_quantization,
      },
    };
  }
  if (stage === "edit_caption") {
    const f = forms.editCaptionForm ?? defaultEditCaptionForm();
    return {
      ...base,
      edit_caption: {
        control_path: f.control_path,
        model: f.model,
        gguf_quantization: f.gguf_quantization,
        prompt: f.prompt,
        overwrite: effectiveOverwrite(f),
        ...writeModeKey(f),
        // Omitted at its default so an edit step that was only opened keeps its saved config.
        ...(f.target_line !== 1 ? { target_line: f.target_line } : {}),
        max_pixels: f.max_pixels,
        max_new_tokens: f.max_new_tokens,
        temperature: f.temperature,
        top_p: f.top_p,
        n_parallel: f.n_parallel,
      },
    };
  }
  if (stage === "quality") {
    const qualityForm = forms.qualityForm ?? defaultQualityForm();
    return {
      ...base,
      quality: {
        metric: qualityForm.metric,
        blur_threshold: qualityForm.blur_threshold,
        min_side: qualityForm.min_side,
        min_detail: qualityForm.min_detail,
        aesthetic_min_label: qualityForm.aesthetic_min_label,
        iqa_model: qualityForm.iqa_model,
        iqa_threshold: qualityForm.iqa_threshold,
        action: (qualityForm.move ? "move" : "report") as "move" | "report",
        output_dir: qualityForm.output_dir,
      },
    };
  }
  if (stage === "index") {
    const indexForm = forms.indexForm ?? defaultIndexForm();
    return { ...base, index: { models: [...indexForm.models] } };
  }
  // clean
  const cleanForm = forms.cleanForm ?? defaultCleanForm();
  return {
    ...base,
    clean: {
      confidence: cleanForm.confidence,
      mask_dilation_px: cleanForm.mask_dilation_px,
      in_place: cleanForm.in_place,
      output_dir: cleanForm.output_dir,
      copy_undetected: cleanForm.copy_undetected,
    },
  };
}

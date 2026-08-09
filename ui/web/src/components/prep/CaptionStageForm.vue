<!--
  Captioning options, extracted verbatim from PrepJobFormView.vue. Owns its own
  registry + prompt-catalogue fetches and the debounced server-side prompt
  preview (POST /prep/caption-prompts/preview), which is the single source of
  truth for the exact text the model receives.

  `promptOptions` / `previewText` / `previewNative` are surfaced as extra
  v-models because the summary panel renders them outside this component.
-->
<template>
  <h3 class="section-title">Captioning options</h3>
  <el-form label-position="top" :disabled="disabled">
    <el-form-item>
      <template #label>
        Model <FieldHelpIcon :field="help('JoyCaption (8B, bf16 ~17 GB) writes free-form captions from a composable instruction prompt. ToriiGate (~5B) is an anime specialist that uses your tag line as grounding — pick it when caption style consistency with Danbooru vocabulary matters more than prompt flexibility.')" />
        <FieldPathTag path="caption.model" />
      </template>
      <el-text v-if="modelsLoading" size="small" type="info">Loading models…</el-text>
      <el-radio-group v-else v-model="model.model" class="model-radio-group">
        <el-radio
          v-for="m in captionModels"
          :key="m.id"
          :value="m.id"
          class="model-radio"
        >
          <span class="model-radio__head">
            <span class="model-radio__name">{{ m.id }}</span>
            <el-tag v-if="m.downloaded" size="small" type="success" effect="plain">downloaded</el-tag>
            <el-tag v-else size="small" type="warning" effect="plain">will download</el-tag>
          </span>
          <span v-if="m.notes" class="model-radio__notes">{{ m.notes }}</span>
        </el-radio>
      </el-radio-group>
    </el-form-item>

    <el-alert
      v-if="model.model === 'toriigate-0.5'"
      type="info"
      :closable="false"
      show-icon
      class="mt-8 mb-12"
      title="ToriiGate uses fixed internal prompt formats: 'Concise' maps to its short format, all other bases map to long. Custom modifiers are appended as extra requirements. Open the prompt preview below to see exactly what the model receives — if the output captions read oddly, that is the place to look first."
    />

    <el-form-item v-if="model.model === 'joycaption-beta-one'">
      <template #label>
        Engine <FieldHelpIcon :field="help('hf runs in-process with transformers (any model). vllm runs an isolated, much faster engine (continuous batching + paged attention) over the whole folder — JoyCaption only. vLLM pins its own torch, so it runs as a separate uv overlay; the first run downloads it and the quantized checkpoint.')" />
        <FieldPathTag path="caption.engine" />
      </template>
      <el-radio-group v-model="model.engine">
        <el-radio value="hf">
          hf
          <el-text size="small" type="info"> (transformers, any model)</el-text>
        </el-radio>
        <el-radio value="vllm">
          vllm
          <el-text size="small" type="info"> (fastest, JoyCaption only)</el-text>
        </el-radio>
      </el-radio-group>
    </el-form-item>

    <el-form-item v-if="model.model === 'toriigate-0.5'">
      <template #label>
        Engine <FieldHelpIcon :field="help('gguf runs ToriiGate through llama.cpp (GPU via Vulkan, no CUDA needed) — much faster than transformers, which the model author calls \'extremely slow\'. The binary and quantized GGUF download on first use.')" />
        <FieldPathTag path="caption.engine" />
      </template>
      <el-radio-group v-model="model.engine">
        <el-radio value="hf">
          hf
          <el-text size="small" type="info"> (transformers, any model)</el-text>
        </el-radio>
        <el-radio value="gguf">
          gguf
          <el-text size="small" type="info"> (fastest, via llama.cpp)</el-text>
        </el-radio>
      </el-radio-group>
    </el-form-item>

    <el-form-item v-if="model.engine !== 'vllm' && model.engine !== 'gguf'">
      <template #label>
        Quantization <FieldHelpIcon :field="help('Controls how model weights are stored in VRAM. Start with bf16; drop to int8 or nf4 if the job fails with an out-of-memory error.')" />
        <FieldPathTag path="caption.quantization" />
      </template>
      <el-radio-group v-model="model.quantization" class="quant-radio-group">
        <el-radio value="bf16">
          bf16
          <el-text size="small" type="info"> (~17 GB JoyCaption / ~10 GB ToriiGate)</el-text>
        </el-radio>
        <el-radio value="int8">
          int8
          <el-text size="small" type="info"> (≈half VRAM — use for JoyCaption on 16 GB cards)</el-text>
        </el-radio>
        <el-radio value="nf4">
          nf4
          <el-text size="small" type="info"> (smallest VRAM — use when int8 still OOMs)</el-text>
        </el-radio>
      </el-radio-group>
    </el-form-item>

    <template v-if="model.engine === 'vllm'">
      <el-form-item>
        <template #label>
          vLLM quantization <FieldHelpIcon :field="help('gptq is a prebuilt 4-bit checkpoint (~5 GB, fits a 16 GB card) and is the fast default. fp8 is data-free (~8.5 GB, no checkpoint). none is full bf16 (~17 GB, needs a 24 GB card). awq has no public checkpoint — set the repo below.')" />
          <FieldPathTag path="caption.vllm_quantization" />
        </template>
        <el-radio-group v-model="model.vllm_quantization" class="quant-radio-group">
          <el-radio value="gptq">gptq <el-text size="small" type="info"> (4-bit, fits 16 GB)</el-text></el-radio>
          <el-radio value="fp8">fp8 <el-text size="small" type="info"> (~8.5 GB, no checkpoint)</el-text></el-radio>
          <el-radio value="none">none <el-text size="small" type="info"> (bf16, 24 GB card)</el-text></el-radio>
          <el-radio value="awq">awq <el-text size="small" type="info"> (needs a repo below)</el-text></el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item>
        <template #label>
          vLLM checkpoint repo <FieldHelpIcon :field="help('Override the HuggingFace repo for the vLLM run. Leave empty to use the default for the chosen quantization. Required for awq (no public checkpoint exists).')" />
          <FieldPathTag path="caption.vllm_model" />
        </template>
        <el-input v-model="model.vllm_model" placeholder="e.g. NeoChen1024/llama-joycaption-beta-one-hf-llava-GPTQ-4bit-sym-autoround" class="w-full" />
      </el-form-item>
    </template>

    <template v-if="model.engine === 'gguf'">
      <el-form-item>
        <template #label>
          GGUF quantization <FieldHelpIcon :field="help('Weight quantization for the llama.cpp run. Q8_0 is effectively lossless (recommended for ToriiGate\'s quality); drop to Q6_K/Q5_K_M/Q4_K_M to trade a little quality for speed and VRAM. The vision projector stays fp16 regardless.')" />
          <FieldPathTag path="caption.gguf_quantization" />
        </template>
        <el-radio-group v-model="model.gguf_quantization" class="quant-radio-group">
          <el-radio value="Q8_0">Q8_0 <el-text size="small" type="info"> (≈ lossless, ~8 GB)</el-text></el-radio>
          <el-radio value="Q6_K">Q6_K <el-text size="small" type="info"> (smaller)</el-text></el-radio>
          <el-radio value="Q5_K_M">Q5_K_M <el-text size="small" type="info"> (~5 GB)</el-text></el-radio>
          <el-radio value="Q4_K_M">Q4_K_M <el-text size="small" type="info"> (smallest/fastest)</el-text></el-radio>
        </el-radio-group>
      </el-form-item>
    </template>

    <el-form-item>
      <template #label>
        Prompt base <FieldHelpIcon :field="help('Sets the core intent of every generated caption. Change it when the default captions are too verbose, or when you need captions centered on a specific aspect (character, style). A custom prompt in the field below overrides the whole composition.')" />
        <FieldPathTag path="caption.prompt_base" />
      </template>
      <el-select v-model="model.prompt_base" class="w-full">
        <el-option
          v-for="base in promptOptions?.bases ?? []"
          :key="base.id"
          :label="base.label"
          :value="base.id"
        />
      </el-select>
      <el-text v-if="activeBase" size="small" type="info" class="preset-desc">
        {{ activeBase.description }}
      </el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        Prompt modifiers (stackable) <FieldHelpIcon :field="help('Each modifier adds one instruction to the base prompt — they compose independently, so tick only what your dataset needs. Check each modifier\'s description and the prompt preview to see the exact effect before running a large job.')" />
        <FieldPathTag path="caption.prompt_modifiers" />
      </template>
      <el-checkbox-group v-model="model.prompt_modifiers">
        <el-checkbox
          v-for="mod in promptOptions?.modifiers ?? []"
          :key="mod.id"
          :value="mod.id"
        >
          {{ mod.label }}
          <el-text size="small" type="info"> — {{ mod.description }}</el-text>
        </el-checkbox>
      </el-checkbox-group>
    </el-form-item>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Character trigger name <FieldHelpIcon :field="help('The model replaces inherent traits (hair, eye color, face) with this name so they collapse into the trigger token at training time. Set it when you want prompting the trigger to reliably reproduce the character\'s fixed appearance without describing those traits explicitly.')" />
          <FieldPathTag path="caption.character_name" />
        </template>
        <el-input
          v-model="model.character_name"
          placeholder="e.g. hatsune miku — inherent traits stay in the name"
          class="w-full"
        />
        <el-text v-if="model.character_name.trim()" size="small" type="info" class="hint-text">
          A scrubber removes any trait clauses the model leaks anyway, keeping absorption consistent. Disabled when a canonical look is set.
        </el-text>
      </el-form-item>
      <el-form-item>
        <template #label>
          Outfit policy (with trigger) <FieldHelpIcon :field="help('Controls whether the outfit is described (stays swappable at generation) or absorbed into the trigger (default outfit always appears). Mixed 50/50 gives the model both signals — use it when you want outfit swapping to work but the canonical look as the default.')" />
          <FieldPathTag path="caption.outfit" />
        </template>
        <el-select
          v-model="model.outfit"
          class="w-full"
          :disabled="!model.character_name.trim()"
        >
          <el-option label="Describe — outfit swappable at gen time" value="describe" />
          <el-option label="Omit — default outfit absorbed into trigger" value="omit" />
          <el-option label="Mixed 50/50 — both signals (recommended)" value="mixed" />
        </el-select>
      </el-form-item>
    </div>

    <el-form-item
      v-if="model.character_name.trim()"
    >
      <template #label>
        Canonical look <FieldHelpIcon :field="help('For datasets with character variants: describe the character\'s baseline appearance (e.g. &quot;aqua twin-tail hair, blue eyes&quot;). Traits that match are absorbed into the trigger; deviations (aged-up versions, alternate hairstyles) are described, keeping them promptable. Use only when your dataset deliberately mixes canon and variant images.')" />
        <FieldPathTag path="caption.character_canon" />
      </template>
      <el-input
        v-model="model.character_canon"
        type="textarea"
        :rows="2"
        placeholder="e.g. aqua twin-tail hair, blue eyes, slim teenage build — deviations (aged-up, alternate hairstyle, meme forms) get described instead of absorbed"
        class="w-full"
      />
      <el-text v-if="model.character_canon.trim()" size="small" type="warning">
        The trait scrubber is disabled in canon mode — if the model fails to separate a variant from canon, those trait clauses will remain in the caption.
      </el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        Caption line <FieldHelpIcon :field="help('Writes the caption to this line in each sidecar file, leaving all other lines untouched. Use 3+ to add a second caption variant — each line is treated as an independent caption at training time.')" />
        <FieldPathTag path="caption.target_line" />
      </template>
      <el-input-number
        v-model="model.target_line"
        :min="2"
        :max="9"
        placeholder="2"
        controls-position="right"
      />
      <el-text size="small" type="info" class="ml-8">
        Line 2 = standard caption. Use 3+ to ADD a caption variant (e.g. queue a
        second job: line 2 trigger-absorbed, line 3 full description).
      </el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        Prompt <FieldHelpIcon :field="help('Auto-filled with the composed prompt (base + modifiers + character settings) and kept in sync as you change those options. Edit the text to send your own wording instead; clear it (or Reset) to go back to the composition.')" />
        <FieldPathTag path="caption.prompt" />
      </template>
      <div class="prompt-state">
        <el-tag size="small" :type="promptDirty ? 'warning' : 'info'" effect="plain">
          {{ promptDirty ? 'Custom (edited)' : 'Composed (auto)' }}
        </el-tag>
        <el-text v-if="!promptDirty && previewNative" size="small" type="info">
          editing leaves ToriiGate's native format
        </el-text>
        <el-button
          v-if="promptDirty"
          link
          type="primary"
          size="small"
          @click="resetPromptToComposed"
        >
          Reset to composed
        </el-button>
      </div>
      <el-input
        :model-value="promptText"
        type="textarea"
        :autosize="{ minRows: 3, maxRows: 24 }"
        placeholder="Composed from the options above — edit to customize"
        class="w-full"
        @update:model-value="onPromptInput"
      />
      <el-text size="small" type="info" class="hint-text">
        The exact text the model receives is shown live in the Summary panel.
      </el-text>
    </el-form-item>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Batch size <FieldHelpIcon :field="help('Images captioned per forward pass (default 4). Raise it to caption faster when VRAM allows; lower it to 1 if the job runs out of memory.')" />
          <FieldPathTag path="caption.batch_size" />
        </template>
        <el-input-number v-model="model.batch_size" :min="1" :max="16" placeholder="4" controls-position="right" class="w-full" />
      </el-form-item>
      <el-form-item>
        <template #label>
          Max new tokens <FieldHelpIcon :field="help('Upper bound on caption length in tokens (default 512). Raise it if long captions get cut off mid-sentence; lower it to keep captions terse and save time.')" />
          <FieldPathTag path="caption.max_new_tokens" />
        </template>
        <el-input-number v-model="model.max_new_tokens" :min="32" :max="4096" placeholder="512" controls-position="right" class="w-full" />
      </el-form-item>
    </div>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Max image side (px, 0 = no downscale) <FieldHelpIcon :field="help('Downscales the long side of each image to this limit before the VLM sees it (default 1536 px). Lower it to reduce VRAM per image when captioning very large originals; training bucketing handles the real resize independently.')" />
          <FieldPathTag path="caption.max_image_side" />
        </template>
        <el-input-number v-model="model.max_image_side" :min="0" :step="128" placeholder="1536" controls-position="right" class="w-full" />
        <el-text v-if="model.model === 'toriigate-0.5'" size="small" type="info" class="hint-text">
          ToriiGate additionally caps inputs at ~1.0 Mpx (its training resolution).
        </el-text>
      </el-form-item>
      <el-form-item>
        <template #label>
          Min image side (px, 0 = no filter) <FieldHelpIcon :field="help('Skips images whose short side is below this limit. Set it when your folder contains thumbnails or web-scrape artifacts that produce garbled captions.')" />
          <FieldPathTag path="caption.min_image_side" />
        </template>
        <el-input-number v-model="model.min_image_side" :min="0" :step="64" placeholder="0" controls-position="right" class="w-full" />
      </el-form-item>
    </div>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Temperature <FieldHelpIcon :field="help('Controls output randomness; leave blank for the recommended value of the selected model (shown in the placeholder). Lower it for more consistent captions across similar images; raise it only if captions feel repetitive.')" />
          <FieldPathTag path="caption.temperature" />
        </template>
        <el-input-number
          v-model="model.temperature"
          :min="0"
          :max="2"
          :step="0.05"
          :precision="2"
          :value-on-clear="null"
          :placeholder="samplingDefaultsPlaceholder('temperature')"
          controls-position="right"
          class="w-full"
        />
      </el-form-item>
      <el-form-item>
        <template #label>
          Top-p <FieldHelpIcon :field="help('Nucleus sampling cutoff — limits which tokens are considered each step; leave blank for the model default (shown in the placeholder). Rarely needs changing; lower it only if captions produce incoherent or off-topic tokens.')" />
          <FieldPathTag path="caption.top_p" />
        </template>
        <el-input-number
          v-model="model.top_p"
          :min="0"
          :max="1"
          :step="0.05"
          :precision="2"
          :value-on-clear="null"
          :placeholder="samplingDefaultsPlaceholder('top_p')"
          controls-position="right"
          class="w-full"
        />
      </el-form-item>
    </div>

    <el-form-item v-if="model.model === 'toriigate-0.5'">
      <template #label>
        Exact generation <FieldHelpIcon :field="help('Generates one image at a time instead of in batches, giving bit-exact results (~2.5x slower). Turn on if batched captions for similar images come out phrased inconsistently.')" />
        <FieldPathTag path="caption.exact_generation" />
      </template>
      <el-switch v-model="model.exact_generation" />
      <el-text class="ml-8" size="small">Exact (unpadded) generation</el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        Use tags as grounding <FieldHelpIcon :field="help('Feeds the line-1 booru tags to ToriiGate as context, improving tag/caption consistency. Turn off only if the tag line is absent or unreliable.')" />
        <FieldPathTag path="caption.use_tags_as_grounding" />
      </template>
      <el-switch v-model="model.use_tags_as_grounding" />
      <el-text class="ml-8" size="small">Use tags as grounding</el-text>
    </el-form-item>

    <el-form-item>
      <template #label>
        Overwrite <FieldHelpIcon :field="help('Re-captions images that already have content on the target line, replacing it. Turn on when you are changing the prompt or model and want to regenerate captions for the whole folder.')" />
        <FieldPathTag path="caption.overwrite" />
      </template>
      <el-switch v-model="model.overwrite" />
      <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import type { PropType } from "vue";
import { api } from "../../api";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import { copyKnown, help } from "./formHelpers";
import type { PrepCaptionForm } from "../../lib/prepStageConfig";
import type { PrepCaptionConfig, PrepModelInfo, PrepPromptOptions } from "../../types/api";

const model = defineModel<PrepCaptionForm>({ required: true });
/** Surfaced upward: the summary panel labels the base/modifiers from this. */
const promptOptions = defineModel<PrepPromptOptions | null>("promptOptions", { default: null });
const previewText = defineModel<string>("previewText", { default: "" });
const previewNative = defineModel<boolean>("previewNative", { default: false });

const props = defineProps({
  /** `caption` section of a cloned job config; applied once the catalogue loaded. */
  seed: { type: Object as PropType<PrepCaptionConfig | null>, default: null },
  /**
   * Read-only. `el-form` hands this to every Element Plus control under it, so one binding
   * disables the whole stage form — the prompt textarea and its Reset button included.
   */
  disabled: { type: Boolean, default: false },
});

const captionModels = ref<PrepModelInfo[]>([]);
const modelsLoading = ref(false);

const activeBase = computed(() =>
  promptOptions.value?.bases.find((b) => b.id === model.value.prompt_base)
);

function samplingDefaultsPlaceholder(field: "temperature" | "top_p"): string {
  const defaults = promptOptions.value?.sampling_defaults?.[model.value.model];
  if (!defaults) return "model default";
  const val = defaults[field];
  if (val == null) return "model default";
  return `model default (${val})`;
}

// --- server-side prompt preview ---
let _previewGen = 0;
let _previewTimer: ReturnType<typeof setTimeout> | null = null;

function captionConfigSnapshot() {
  return {
    model: model.value.model,
    quantization: model.value.quantization,
    prompt: model.value.prompt,
    prompt_base: model.value.prompt_base,
    prompt_modifiers: [...model.value.prompt_modifiers],
    character_name: model.value.character_name,
    character_canon: model.value.character_canon,
    outfit: model.value.outfit,
    target_line: model.value.target_line,
    max_new_tokens: model.value.max_new_tokens,
    temperature: model.value.temperature,
    top_p: model.value.top_p,
    exact_generation: model.value.exact_generation,
    batch_size: model.value.batch_size,
    use_tags_as_grounding: model.value.use_tags_as_grounding,
    overwrite: model.value.overwrite,
    max_image_side: model.value.max_image_side,
    min_image_side: model.value.min_image_side,
    engine: model.value.engine,
    vllm_quantization: model.value.vllm_quantization,
    vllm_model: model.value.vllm_model,
    gguf_quantization: model.value.gguf_quantization,
  };
}

function schedulePreview() {
  if (_previewTimer !== null) clearTimeout(_previewTimer);
  _previewTimer = setTimeout(async () => {
    _previewTimer = null;
    const gen = ++_previewGen;
    try {
      const result = await api.prepCaptionPromptPreview(captionConfigSnapshot());
      if (gen !== _previewGen) return; // stale
      previewText.value = result.prompt;
      previewNative.value = result.native_format;
    } catch {
      // preview is best-effort; don't surface errors
    }
  }, 400);
}

watch(
  () => [
    model.value.model,
    model.value.prompt,
    model.value.prompt_base,
    model.value.prompt_modifiers.slice(),
    model.value.character_name,
    model.value.character_canon,
    model.value.outfit,
    model.value.use_tags_as_grounding,
  ],
  () => schedulePreview(),
  { deep: true }
);

// Prompt editing: the textarea mirrors the live composed preview until the user
// edits it; from then on its text is the custom override (caption.prompt) and the
// auto-sync stops. Clearing the field (or Reset) returns to the composition.
const promptText = ref("");
const promptDirty = ref(false);
const composedText = ref("");

watch(previewText, (v) => {
  if (!promptDirty.value) {
    composedText.value = v;
    promptText.value = v;
  }
});

function onPromptInput(val: string): void {
  promptText.value = val;
  if (val.trim()) {
    promptDirty.value = true;
    model.value.prompt = val;
  } else {
    promptDirty.value = false;
    model.value.prompt = "";
  }
}

function resetPromptToComposed(): void {
  promptDirty.value = false;
  model.value.prompt = "";
  promptText.value = composedText.value;
}

const loaded = ref(false);
let seedApplied = false;

/** Seed from a cloned job config; a non-empty prompt is a custom override. */
function applySeed(): void {
  const seed = props.seed;
  if (!seed || seedApplied) return;
  seedApplied = true;
  copyKnown(model.value as unknown as Record<string, unknown>, seed);
  if (model.value.prompt.trim()) {
    promptDirty.value = true;
    promptText.value = model.value.prompt;
  }
}

async function loadModels(): Promise<void> {
  modelsLoading.value = true;
  try {
    const res = await api.prepModels("caption");
    captionModels.value = res.models || [];
    const first = captionModels.value[0];
    // The registry's picks fill GAPS, they never replace a choice: `model` is a shared v-model,
    // and in the workflow drawer a write here is an edit the parent saves. A seeded form already
    // carries the user's own model and prompt, and those must survive being looked at.
    if (first && !model.value.model) model.value.model = first.id;
    const prompts = await api.prepCaptionPrompts();
    promptOptions.value = prompts;
    if (!seedApplied) {
      if (prompts.default_base) model.value.prompt_base = prompts.default_base;
      if (prompts.default_modifiers) model.value.prompt_modifiers = [...prompts.default_modifiers];
    }
  } catch {
    // models endpoint may not be implemented yet — silently degrade
  } finally {
    modelsLoading.value = false;
    loaded.value = true;
    // A seed that arrived while the catalogues were in flight is applied now (the watcher below
    // ignores it until `loaded`); `applySeed` is one-shot, so a seed already applied is a no-op.
    applySeed();
  }
}

// A clone seed overrides the loaded defaults, whichever settles last.
watch(
  () => props.seed,
  () => {
    if (loaded.value) applySeed();
  }
);

/**
 * Seed first, and **synchronously**.
 *
 * `loadModels` awaits, and Vue flushes watchers in the microtask that await opens — so seeding
 * after it means the parent sees one tick of registry-shaped state and takes it for an edit.
 * In the workflow drawer that tick reaches `watch(builtConfig)`, which emits `update:node`, and
 * merely *opening* a saved step autosaves the registry's guess over its config.
 */
onMounted(() => {
  applySeed();
  void loadModels();
});
</script>

<style scoped>
.section-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.form-row-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
@media (max-width: 600px) {
  .form-row-2 {
    grid-template-columns: 1fr;
  }
}
.model-radio-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}
.model-radio {
  height: auto;
  width: 100%;
  align-items: flex-start;
  margin-right: 0;
}
.model-radio :deep(.el-radio__label) {
  display: flex;
  flex-direction: column;
  gap: 2px;
  white-space: normal;
  line-height: 1.4;
}
.model-radio__head {
  display: flex;
  align-items: center;
  gap: 8px;
}
.model-radio__name {
  font-weight: 500;
}
.model-radio__notes {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.quant-radio-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}
.preset-desc {
  margin-top: 4px;
  display: block;
}
.prompt-state {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.hint-text {
  display: block;
  margin-top: 4px;
  margin-bottom: 8px;
}
.ml-8 {
  margin-left: 8px;
}
.mt-8 {
  margin-top: 8px;
}
</style>

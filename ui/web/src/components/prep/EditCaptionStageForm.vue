<!--
  Edit-instruction captioning options (`[edit_caption]`), in the style of
  CaptionStageForm: owns the GGUF model registry fetch and the edit-prompt
  catalogue (GET /prep/edit-caption-prompts), whose default prompt is shown in
  the textarea until the user edits it.

  `previewText` is surfaced as an extra v-model: the summary panel renders the
  full request layout (image slots + prompt) outside this component.
-->
<template>
  <h3 class="section-title">Edit instruction options</h3>
  <el-alert
    type="warning"
    :closable="false"
    show-icon
    class="mb-12"
    title="VLMs hallucinate differences between two images (EditCaption, arXiv 2604.08213): some instructions will name changes that are not there or miss the real one. Review them before training: Studio → Tag editor shows each target's line 1; fix wrong ones in its caption file."
  />
  <el-form label-position="top" :disabled="disabled">
    <el-form-item required>
      <template #label>
        Control folder <FieldHelpIcon :field="help('Folder with the control (source) images. Each target in the dataset folder pairs with stem.<ext> (one control) or stem_0.<ext>, stem_1.<ext>, … (several, in order) — the same rule the trainer uses for control_path. Targets without a valid set are listed as unpaired in the report and left untouched.')" />
        <FieldPathTag path="edit_caption.control_path" />
      </template>
      <PathFieldControl
        v-model="model.control_path"
        expect="dir"
        required
        placeholder="e.g. /path/to/dataset/controls"
        input-class="w-full"
      />
    </el-form-item>

    <el-form-item>
      <template #label>
        Model <FieldHelpIcon :field="help('Qwen3-VL Instruct through llama.cpp (GGUF, GPU via Vulkan). The 4B model fits an 8 GB card; the 8B writes better instructions and needs ~7.5 GB at Q4_K_M. The binary and the weights download on first use.')" />
        <FieldPathTag path="edit_caption.model" />
      </template>
      <el-text v-if="modelsLoading" size="small" type="info">Loading models…</el-text>
      <el-radio-group v-else v-model="model.model" class="model-radio-group">
        <el-radio v-for="m in models" :key="m.id" :value="m.id" class="model-radio">
          <span class="model-radio__head">
            <span class="model-radio__name">{{ m.id }}</span>
            <el-tag v-if="m.downloaded" size="small" type="success" effect="plain">downloaded</el-tag>
            <el-tag v-else size="small" type="warning" effect="plain">will download</el-tag>
          </span>
          <span v-if="m.notes" class="model-radio__notes">{{ m.notes }}</span>
        </el-radio>
      </el-radio-group>
    </el-form-item>

    <el-form-item>
      <template #label>
        GGUF quantization <FieldHelpIcon :field="help('Weight quantization for the llama.cpp run. Model default is Q8_0 for the 4B and Q4_K_M for the 8B (what fits an 8 GB card). The vision projector stays fp16 regardless.')" />
        <FieldPathTag path="edit_caption.gguf_quantization" />
      </template>
      <el-radio-group v-model="model.gguf_quantization" class="quant-radio-group">
        <el-radio value="">
          model default
          <el-text v-if="activeModel?.default_quant" size="small" type="info"> ({{ activeModel.default_quant }})</el-text>
        </el-radio>
        <el-radio v-for="q in QUANTS" :key="q" :value="q">{{ q }}</el-radio>
      </el-radio-group>
    </el-form-item>

    <el-form-item>
      <template #label>
        Prompt <FieldHelpIcon :field="help('The instruction the model follows, shown with its default text. The image labels (Source / Result, or Image 1, Image 2 … for several controls) are added around it automatically — see the request layout in the Summary panel. Edit to customize; clear it (or Reset) to go back to the default.')" />
        <FieldPathTag path="edit_caption.prompt" />
      </template>
      <div class="prompt-state">
        <el-tag size="small" :type="promptDirty ? 'warning' : 'info'" effect="plain">
          {{ promptDirty ? 'Custom (edited)' : 'Default' }}
        </el-tag>
        <el-button v-if="promptDirty" link type="primary" size="small" @click="resetPrompt">
          Reset to default
        </el-button>
      </div>
      <el-input
        :model-value="promptText"
        type="textarea"
        :autosize="{ minRows: 3, maxRows: 16 }"
        placeholder="Default edit-instruction prompt"
        class="w-full"
        @update:model-value="onPromptInput"
      />
      <el-text size="small" type="info" class="hint-text">
        The reply is written to line 1 of each target's caption, with quotes and labels like "Instruction:" stripped.
      </el-text>
    </el-form-item>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Max pixels per image <FieldHelpIcon :field="help('Each control and the target are downscaled to at most this many pixels before the VLM sees them (default 524288 ≈ 724×724, ~512 tokens per image). The server context is sized from it and the images per row, so raising it costs VRAM; lower it if the server fails to start on a small card.')" />
          <FieldPathTag path="edit_caption.max_pixels" />
        </template>
        <el-input-number v-model="model.max_pixels" :min="65536" :step="65536" controls-position="right" class="w-full" />
      </el-form-item>
      <el-form-item>
        <template #label>
          Parallel slots (0 = model default) <FieldHelpIcon :field="help('Requests llama-server processes at once (model default 4). More slots is faster but multiplies the context (VRAM); drop to 1–2 on a card that runs out of memory.')" />
          <FieldPathTag path="edit_caption.n_parallel" />
        </template>
        <el-input-number v-model="model.n_parallel" :min="0" :max="16" controls-position="right" class="w-full" />
      </el-form-item>
    </div>

    <div class="form-row-2">
      <el-form-item>
        <template #label>
          Temperature <FieldHelpIcon :field="help('Sampling randomness (default 0.2: consistent, literal instructions). Clear it to use the model\'s recommended 0.7.')" />
          <FieldPathTag path="edit_caption.temperature" />
        </template>
        <el-input-number
          v-model="model.temperature"
          :min="0"
          :max="2"
          :step="0.05"
          :precision="2"
          :value-on-clear="null"
          placeholder="model default (0.7)"
          controls-position="right"
          class="w-full"
        />
      </el-form-item>
      <el-form-item>
        <template #label>
          Max new tokens <FieldHelpIcon :field="help('Upper bound on the instruction length (default 96 — one or two sentences). Raise it only if instructions get cut off.')" />
          <FieldPathTag path="edit_caption.max_new_tokens" />
        </template>
        <el-input-number v-model="model.max_new_tokens" :min="16" :max="512" controls-position="right" class="w-full" />
      </el-form-item>
    </div>

    <el-form-item>
      <template #label>
        Overwrite <FieldHelpIcon :field="help('Rewrites targets whose line 1 already has text. Off (default) skips them, so a stopped job resumes where it left off.')" />
        <FieldPathTag path="edit_caption.overwrite" />
      </template>
      <el-switch v-model="model.overwrite" />
      <el-text class="ml-8" size="small">Overwrite existing instructions</el-text>
    </el-form-item>
  </el-form>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import type { PropType } from "vue";
import { api } from "../../api";
import FieldHelpIcon from "../FieldHelpIcon.vue";
import FieldPathTag from "../FieldPathTag.vue";
import PathFieldControl from "../PathFieldControl.vue";
import { copyKnown, help } from "./formHelpers";
import type { PrepEditCaptionForm } from "../../lib/prepStageConfig";
import type { PrepEditCaptionConfig, PrepModelInfo } from "../../types/api";

const QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"] as const;

const model = defineModel<PrepEditCaptionForm>({ required: true });
/** Surfaced upward: the summary panel shows the full request layout. */
const previewText = defineModel<string>("previewText", { default: "" });

const props = defineProps({
  /** `edit_caption` section of a cloned job config; applied once the registry loaded. */
  seed: { type: Object as PropType<PrepEditCaptionConfig | null>, default: null },
  /** Read-only: disables every control of the form. */
  disabled: { type: Boolean, default: false },
});

const models = ref<PrepModelInfo[]>([]);
const modelsLoading = ref(false);
const activeModel = computed(() => models.value.find((m) => m.id === model.value.model));

// Prompt editing: the textarea shows the default until edited; from then on its text is
// the custom override (edit_caption.prompt). Clearing it (or Reset) returns to the default.
const defaultPrompt = ref("");
const promptText = ref("");
const promptDirty = ref(false);

function onPromptInput(val: string): void {
  promptText.value = val;
  if (val.trim() && val.trim() !== defaultPrompt.value.trim()) {
    promptDirty.value = true;
    model.value.prompt = val;
  } else {
    promptDirty.value = false;
    model.value.prompt = "";
  }
}

function resetPrompt(): void {
  promptDirty.value = false;
  model.value.prompt = "";
  promptText.value = defaultPrompt.value;
}

// --- request layout preview (server-rendered, debounced) ---
let _gen = 0;
let _timer: ReturnType<typeof setTimeout> | null = null;

function schedulePreview(): void {
  if (_timer !== null) clearTimeout(_timer);
  _timer = setTimeout(async () => {
    _timer = null;
    const gen = ++_gen;
    try {
      const res = await api.prepEditCaptionPrompts(model.value.prompt);
      if (gen !== _gen) return; // stale
      if (!defaultPrompt.value) {
        defaultPrompt.value = res.default_prompt;
        if (!promptDirty.value) promptText.value = res.default_prompt;
      }
      previewText.value = `${res.layout_single}\n\n— with two controls —\n${res.layout_multi}`;
    } catch {
      // preview is best-effort
    }
  }, 300);
}

watch(() => model.value.prompt, () => schedulePreview());

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

watch(
  () => props.seed,
  () => applySeed()
);

async function loadModels(): Promise<void> {
  modelsLoading.value = true;
  try {
    const res = await api.prepModels("edit_caption");
    models.value = res.models || [];
    const first = models.value[0];
    // Fill a gap only — never replace a seeded or already chosen model.
    if (first && !model.value.model) model.value.model = first.id;
  } catch {
    // registry unavailable — the user can still submit a seeded config
  } finally {
    modelsLoading.value = false;
  }
}

onMounted(() => {
  applySeed(); // synchronously, before the awaits (see CaptionStageForm)
  schedulePreview();
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
.model-radio-group,
.quant-radio-group {
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
.prompt-state {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.hint-text {
  display: block;
  margin-top: 4px;
}
.ml-8 {
  margin-left: 8px;
}
.mb-12 {
  margin-bottom: 12px;
}
</style>

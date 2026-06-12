<template>
  <div class="prep-form-page page-shell">
    <div class="page-head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Dataset Studio</el-button>
      <span class="prep-form-title">New {{ stageLabel }} job</span>
    </div>

    <el-alert v-if="formError" type="error" :title="formError" show-icon class="mt-12" />

    <el-card shadow="never" class="mt-12">
      <div class="prep-form-body">
        <!-- Common fields -->
        <el-form label-position="top">
          <el-form-item required>
            <template #label>
              Dataset folder <FieldHelpIcon :field="help('Path to the image folder to process. Only images directly inside the folder are scanned (no subfolders — same rule as training).')" />
            </template>
            <PathFieldControl
              v-model="form.path"
              expect="dir"
              required
              placeholder="/path/to/dataset"
              input-class="w-full"
            />
          </el-form-item>

          <div class="form-row-2">
            <el-form-item>
              <template #label>
                Caption format <FieldHelpIcon :field="help('Sidecar: one .txt per image, each line a caption variant (line 1 = tags, line 2 = caption). JSON: single captions.json index file per folder.')" />
              </template>
              <el-select v-model="form.caption_format" class="w-full">
                <el-option label="Caption files (.txt next to each image)" value="sidecar" />
                <el-option label="captions.json (single index file)" value="json" />
              </el-select>
            </el-form-item>
            <el-form-item v-if="form.caption_format !== 'json'" label="Caption extension">
              <el-input v-model="form.caption_ext" placeholder=".txt" class="w-full" />
            </el-form-item>
          </div>
        </el-form>

        <el-divider />

        <!-- Tag stage -->
        <template v-if="stage === 'tag'">
          <h3 class="section-title">Tagging options</h3>
          <el-form label-position="top">
            <el-form-item>
              <template #label>
                Models <FieldHelpIcon :field="help('ONNX tagger ensemble — probabilities merge across models by max. Pre-selects downloaded models.')" />
              </template>
              <el-text v-if="modelsLoading" size="small" type="info">Loading models…</el-text>
              <el-select
                v-else
                v-model="tagForm.models"
                multiple
                filterable
                placeholder="Select tagger models"
                class="w-full"
              >
                <el-option
                  v-for="m in tagModels"
                  :key="m.id"
                  :label="`${m.id}${m.downloaded ? ' ✓' : ' (will download)'}`"
                  :value="m.id"
                >
                  <span>{{ m.id }}</span>
                  <el-tag v-if="m.downloaded" size="small" type="success" effect="plain" class="ml-8">downloaded</el-tag>
                  <el-tag v-else size="small" type="warning" effect="plain" class="ml-8">will download</el-tag>
                </el-option>
              </el-select>
            </el-form-item>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  General tag confidence <FieldHelpIcon :field="help('Minimum probability to include a general (non-character, non-rating) tag. Leave blank to use the model default.')" />
                </template>
                <el-input-number
                  v-model="tagForm.general_threshold"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  :precision="2"
                  :value-on-clear="null"
                  controls-position="right"
                  placeholder="model default"
                  class="w-full"
                />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Character tag confidence <FieldHelpIcon :field="help('Minimum probability for character/series tags. Taggers are weakest here — a higher threshold keeps only confident character matches.')" />
                </template>
                <el-input-number
                  v-model="tagForm.character_threshold"
                  :min="0"
                  :max="1"
                  :step="0.05"
                  :precision="2"
                  :value-on-clear="null"
                  controls-position="right"
                  placeholder="model default"
                  class="w-full"
                />
              </el-form-item>
            </div>
            <el-text size="small" type="info" class="hint-text">
              Higher = fewer but surer tags. Output tags are ordered by confidence (most certain first).
            </el-text>

            <el-form-item class="mt-8">
              <el-switch v-model="tagForm.include_character_tags" />
              <el-text class="ml-8" size="small">
                Include character/series name tags
                <el-text type="info"> — taggers are weakest at character names; disable to keep only your own trigger via Prepend tags</el-text>
              </el-text>
            </el-form-item>

            <el-form-item>
              <el-switch v-model="tagForm.include_rating" />
              <el-text class="ml-8" size="small">Include rating tag (general/sensitive/questionable/explicit)</el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Exclude tags <FieldHelpIcon :field="help('Tags to strip from the output — even if the model is confident about them.')" />
              </template>
              <el-select
                v-model="tagForm.exclude_tags"
                multiple
                filterable
                allow-create
                default-first-option
                placeholder="Tags to exclude from output"
                class="w-full"
              >
                <el-option v-for="t in tagForm.exclude_tags" :key="t" :label="t" :value="t" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <template #label>
                Prepend tags <FieldHelpIcon :field="help('Tags prepended to every output caption — use for trigger words or forced tags the tagger may miss.')" />
              </template>
              <el-select
                v-model="tagForm.prepend_tags"
                multiple
                filterable
                allow-create
                default-first-option
                placeholder="Tags to prepend to every caption"
                class="w-full"
              >
                <el-option v-for="t in tagForm.prepend_tags" :key="t" :label="t" :value="t" />
              </el-select>
            </el-form-item>

            <div class="form-row-2">
              <el-form-item label="Max tags">
                <el-input-number v-model="tagForm.max_tags" :min="1" :max="500" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item label="Batch size">
                <el-input-number v-model="tagForm.batch_size" :min="1" :max="64" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <el-form-item>
              <template #label>
                Overwrite <FieldHelpIcon :field="help('Re-tag images that already have a tag line on line 1. Off by default — skips already-processed images.')" />
              </template>
              <el-switch v-model="tagForm.overwrite" />
              <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
            </el-form-item>

            <el-form-item>
              <el-switch v-model="chainCaption" />
              <el-text class="ml-8" size="small">
                Also queue a caption job after tagging (uses the caption defaults;
                ToriiGate/JoyCaption can be tuned by queueing it separately)
              </el-text>
            </el-form-item>
          </el-form>
        </template>

        <!-- Caption stage -->
        <template v-if="stage === 'caption'">
          <h3 class="section-title">Captioning options</h3>
          <el-form label-position="top">
            <el-form-item>
              <template #label>
                Model <FieldHelpIcon :field="help('JoyCaption: 8B LLaVA, best quality. ToriiGate: ~5B anime specialist with tag grounding.')" />
              </template>
              <el-text v-if="modelsLoading" size="small" type="info">Loading models…</el-text>
              <el-radio-group v-else v-model="captionForm.model" class="model-radio-group">
                <el-radio
                  v-for="m in captionModels"
                  :key="m.id"
                  :value="m.id"
                  class="model-radio"
                >
                  <span>{{ m.id }}</span>
                  <el-tag v-if="m.downloaded" size="small" type="success" effect="plain" class="ml-8">downloaded</el-tag>
                  <el-tag v-else size="small" type="warning" effect="plain" class="ml-8">will download</el-tag>
                  <el-text v-if="m.notes" size="small" type="info" class="ml-8" truncated>{{ m.notes }}</el-text>
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item>
              <template #label>
                Quantization <FieldHelpIcon :field="help('Precision for model weights. Lower precision uses less VRAM at a slight quality cost.')" />
              </template>
              <el-radio-group v-model="captionForm.quantization" class="quant-radio-group">
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
                  <el-text size="small" type="info"> (smallest VRAM, slight quality cost)</el-text>
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item>
              <template #label>
                Prompt base <FieldHelpIcon :field="help('Foundation prompt style: descriptive-long (default), concise, character-focus, style-focus. A custom prompt overrides the whole composition.')" />
              </template>
              <el-select v-model="captionForm.prompt_base" class="w-full">
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
                Prompt modifiers (stackable) <FieldHelpIcon :field="help('Stack onto the base prompt: demographics, medium_neutral (hides style), plain_language, objective_only, composition_camera, explicit_language.')" />
              </template>
              <el-checkbox-group v-model="captionForm.prompt_modifiers">
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
                  Character trigger name (optional) <FieldHelpIcon :field="help('When set, the model refers to the character by this name and absorbs their inherent traits (hair, eyes) into the trigger token.')" />
                </template>
                <el-input
                  v-model="captionForm.character_name"
                  placeholder="e.g. hatsune miku — inherent traits stay in the name"
                  class="w-full"
                />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Outfit policy (with trigger) <FieldHelpIcon :field="help('describe = outfit stays promptable; omit = default outfit absorbed into trigger; mixed = 50/50 per image (recommended).')" />
                </template>
                <el-select
                  v-model="captionForm.outfit"
                  class="w-full"
                  :disabled="!captionForm.character_name.trim()"
                >
                  <el-option label="Describe — outfit swappable at gen time" value="describe" />
                  <el-option label="Omit — default outfit absorbed into trigger" value="omit" />
                  <el-option label="Mixed 50/50 — both signals (recommended)" value="mixed" />
                </el-select>
              </el-form-item>
            </div>

            <el-form-item
              v-if="captionForm.character_name.trim()"
            >
              <template #label>
                Canonical look (optional — for datasets with character variants) <FieldHelpIcon :field="help('Describe the canonical appearance. Deviations from this are described; matches are absorbed into the trigger.')" />
              </template>
              <el-input
                v-model="captionForm.character_canon"
                type="textarea"
                :rows="2"
                placeholder="e.g. aqua twin-tail hair, blue eyes, slim teenage build — deviations (aged-up, alternate hairstyle, meme forms) get described instead of absorbed"
                class="w-full"
              />
              <el-text v-if="captionForm.character_canon.trim()" size="small" type="warning">
                Canon mode trusts the model to separate canon from deviation; the hard
                trait scrubber is disabled.
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Caption line <FieldHelpIcon :field="help('Line index to write the caption to. Line 2 = standard. Use 3+ to add additional caption variants alongside existing ones.')" />
              </template>
              <el-input-number
                v-model="captionForm.target_line"
                :min="2"
                :max="9"
                controls-position="right"
              />
              <el-text size="small" type="info" class="ml-8">
                Line 2 = standard caption. Use 3+ to ADD a caption variant (e.g. queue a
                second job: line 2 trigger-absorbed, line 3 full description).
              </el-text>
            </el-form-item>

            <el-form-item label="Custom prompt (overrides the composition)">
              <el-input
                v-model="captionForm.prompt"
                type="textarea"
                :rows="3"
                :placeholder="composedPreview || 'model default'"
                class="w-full"
              />
            </el-form-item>

            <div class="form-row-2">
              <el-form-item label="Batch size">
                <el-input-number v-model="captionForm.batch_size" :min="1" :max="16" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item label="Max new tokens">
                <el-input-number v-model="captionForm.max_new_tokens" :min="32" :max="4096" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Max image side (px, 0 = no downscale) <FieldHelpIcon :field="help('Downscale images longer than this before sending to the VLM. Default 1536. Bucketing handles the real resize at training time.')" />
                </template>
                <el-input-number v-model="captionForm.max_image_side" :min="0" :step="128" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Min image side (px, 0 = no filter) <FieldHelpIcon :field="help('Skip images with either side below this threshold. Use to filter out tiny thumbnails before captioning.')" />
                </template>
                <el-input-number v-model="captionForm.min_image_side" :min="0" :step="64" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item label="Temperature">
                <el-input-number v-model="captionForm.temperature" :min="0" :max="2" :step="0.05" :precision="2" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item label="Top-p">
                <el-input-number v-model="captionForm.top_p" :min="0" :max="1" :step="0.05" :precision="2" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <el-form-item>
              <el-switch v-model="captionForm.use_tags_as_grounding" />
              <el-text class="ml-8" size="small">
                Use tags as grounding
                <el-text type="info"> (ToriiGate: line-1 tags used as context)</el-text>
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Overwrite <FieldHelpIcon :field="help('Re-caption images that already have content on the target line. Off by default — skips already-processed images.')" />
              </template>
              <el-switch v-model="captionForm.overwrite" />
              <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
            </el-form-item>
          </el-form>
        </template>

        <!-- Clean stage -->
        <template v-if="stage === 'clean'">
          <h3 class="section-title">Cleaning options</h3>
          <el-alert
            type="info"
            :closable="false"
            show-icon
            class="mt-8 mb-12"
            title="Watermark cleanup: a YOLO11 detector finds watermarks and signatures, then LaMa inpainting removes them. Non-destructive by default — cleaned copies are written to the output folder."
          />
          <el-form label-position="top">
            <el-form-item>
              <template #label>
                Confidence threshold <FieldHelpIcon :field="help('Minimum YOLO11 detection score to treat a region as a watermark. Lower = more aggressive removal.')" />
              </template>
              <el-slider
                v-model="cleanForm.confidence"
                :min="0"
                :max="1"
                :step="0.01"
                show-input
                :show-input-controls="false"
              />
            </el-form-item>

            <el-form-item>
              <template #label>
                Mask dilation (px) <FieldHelpIcon :field="help('Pixels to expand the detected watermark mask before inpainting. Larger = covers edge fringing.')" />
              </template>
              <el-input-number v-model="cleanForm.mask_dilation_px" :min="0" :max="100" controls-position="right" />
            </el-form-item>

            <el-form-item>
              <template #label>
                In-place <FieldHelpIcon :field="help('Overwrite originals instead of writing to a separate folder. Originals are backed up under .rengu_prep/cleanup_originals/ first.')" />
              </template>
              <el-switch v-model="cleanForm.in_place" />
              <el-text class="ml-8" size="small">In-place (overwrite originals)</el-text>
              <el-alert
                v-if="cleanForm.in_place"
                type="warning"
                show-icon
                :closable="false"
                class="mt-8"
                title="Originals are backed up under .rengu_prep/ before cleaning"
              />
            </el-form-item>

            <el-form-item v-if="!cleanForm.in_place">
              <template #label>
                Output directory <FieldHelpIcon :field="help('Where to write cleaned images. Defaults to <dataset>/cleaned/.')" />
              </template>
              <PathFieldControl
                v-model="cleanForm.output_dir"
                expect="dir"
                placeholder="<dataset>/cleaned"
                input-class="w-full"
              />
            </el-form-item>

            <el-form-item>
              <el-switch v-model="cleanForm.copy_undetected" />
              <el-text class="ml-8" size="small">Copy images with no detections to output</el-text>
            </el-form-item>
          </el-form>
        </template>

        <!-- Submit actions -->
        <div class="form-actions">
          <el-button @click="$router.push('/prep')">Cancel</el-button>
          <el-button :loading="submitting" @click="submit(false)">Queue</el-button>
          <el-button type="primary" :loading="submitting" @click="submit(true)">Start now</el-button>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import FieldHelpIcon from "../components/FieldHelpIcon.vue";
import PathFieldControl from "../components/PathFieldControl.vue";
import { formatError } from "../lib/formatError";
import type { PrepModelInfo, PrepPromptOptions, PrepStage } from "../types/api";

function help(text: string) {
  return { path: "", type: "string", help: text, doc_path: "user/dataset-prep.md" };
}

const route = useRoute();
const router = useRouter();

const stage = computed(() => (route.params.stage as PrepStage) || "tag");
const stageLabel = computed(() => {
  const map: Record<string, string> = { tag: "tag", caption: "caption", clean: "clean" };
  return map[stage.value] ?? stage.value;
});

// --- common form state ---
const form = reactive({
  path: "",
  caption_format: "sidecar" as "sidecar" | "json",
  caption_ext: ".txt",
});

// --- tag form ---
const tagForm = reactive({
  models: [] as string[],
  exclude_tags: [] as string[],
  prepend_tags: [] as string[],
  max_tags: 40,
  batch_size: 8,
  overwrite: false,
  general_threshold: null as number | null,
  character_threshold: null as number | null,
  include_character_tags: true,
  include_rating: true,
});

// --- caption form ---
const captionForm = reactive({
  model: "",
  quantization: "bf16" as "bf16" | "int8" | "nf4",
  prompt: "",
  prompt_base: "descriptive-long",
  prompt_modifiers: ["demographics"] as string[],
  character_name: "",
  character_canon: "",
  outfit: "describe" as "describe" | "omit" | "mixed",
  target_line: 2,
  max_new_tokens: 512,
  temperature: 0.6,
  top_p: 0.9,
  batch_size: 4,
  use_tags_as_grounding: true,
  overwrite: false,
  max_image_side: 1536,
  min_image_side: 0,
});

// --- clean form ---
const cleanForm = reactive({
  confidence: 0.35,
  mask_dilation_px: 4,
  in_place: false,
  output_dir: "",
  copy_undetected: true,
});

// --- models ---
const tagModels = ref<PrepModelInfo[]>([]);
const captionModels = ref<PrepModelInfo[]>([]);
const modelsLoading = ref(false);
const promptOptions = ref<PrepPromptOptions | null>(null);
const activeBase = computed(() =>
  promptOptions.value?.bases.find((b) => b.id === captionForm.prompt_base)
);
const composedPreview = computed(() => {
  const opts = promptOptions.value;
  if (!opts || !activeBase.value) return "";
  const parts = [activeBase.value.prompt];
  for (const mod of opts.modifiers) {
    if (captionForm.prompt_modifiers.includes(mod.id)) parts.push(mod.text);
  }
  const name = captionForm.character_name.trim();
  if (name) {
    parts.push(opts.character_trigger_template.replace("{name}", name));
    const outfitText =
      captionForm.outfit === "mixed"
        ? "(per-image 50/50: describe or omit the outfit)"
        : opts.outfit_texts[captionForm.outfit] ?? "";
    if (outfitText) parts.push(outfitText);
  }
  parts.push(opts.no_meta);
  return parts.join(" ");
});

async function loadModels(): Promise<void> {
  modelsLoading.value = true;
  try {
    if (stage.value === "tag") {
      const res = await api.prepModels("tag");
      tagModels.value = res.models || [];
      // pre-select downloaded models; fall back to the registry's default ensemble
      tagForm.models = tagModels.value.filter((m) => m.downloaded).map((m) => m.id);
      if (!tagForm.models.length) {
        tagForm.models = tagModels.value.slice(0, 2).map((m) => m.id);
      }
    } else if (stage.value === "caption") {
      const res = await api.prepModels("caption");
      captionModels.value = res.models || [];
      const first = captionModels.value[0];
      if (first) captionForm.model = first.id;
      const prompts = await api.prepCaptionPrompts();
      promptOptions.value = prompts;
      if (prompts.default_base) captionForm.prompt_base = prompts.default_base;
      if (prompts.default_modifiers) captionForm.prompt_modifiers = [...prompts.default_modifiers];
    }
  } catch {
    // models endpoint may not be implemented yet — silently degrade
  } finally {
    modelsLoading.value = false;
  }
}

// --- submit ---
const submitting = ref(false);
const chainCaption = ref(false);
const formError = ref("");

function buildConfig() {
  const base = {
    path: form.path,
    caption_format: form.caption_format,
    caption_ext: form.caption_ext || ".txt",
  };

  if (stage.value === "tag") {
    return {
      ...base,
      tag: {
        models: [...tagForm.models],
        exclude_tags: [...tagForm.exclude_tags],
        prepend_tags: [...tagForm.prepend_tags],
        max_tags: tagForm.max_tags,
        batch_size: tagForm.batch_size,
        overwrite: tagForm.overwrite,
        general_threshold: tagForm.general_threshold,
        character_threshold: tagForm.character_threshold,
        include_character_tags: tagForm.include_character_tags,
        include_rating: tagForm.include_rating,
      },
    };
  }
  if (stage.value === "caption") {
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
        batch_size: captionForm.batch_size,
        use_tags_as_grounding: captionForm.use_tags_as_grounding,
        overwrite: captionForm.overwrite,
        max_image_side: captionForm.max_image_side,
        min_image_side: captionForm.min_image_side,
      },
    };
  }
  // clean
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

async function submit(startNow: boolean): Promise<void> {
  formError.value = "";
  if (!form.path.trim()) {
    formError.value = "Dataset folder is required.";
    return;
  }
  if (stage.value === "tag" && !tagForm.models.length) {
    formError.value = "Select at least one tagger model.";
    return;
  }
  if (stage.value === "caption" && !captionForm.model) {
    formError.value = "Select a caption model.";
    return;
  }
  submitting.value = true;
  try {
    await api.createPrepJob({
      stage: stage.value,
      config: buildConfig(),
      start_now: startNow,
    });
    if (stage.value === "tag" && chainCaption.value) {
      // FIFO queue: the caption job starts automatically when tagging finishes.
      await api.createPrepJob({
        stage: "caption",
        config: {
          path: form.path,
          caption_format: form.caption_format,
          caption_ext: form.caption_ext,
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
            batch_size: captionForm.batch_size,
            use_tags_as_grounding: captionForm.use_tags_as_grounding,
            overwrite: captionForm.overwrite,
            max_image_side: captionForm.max_image_side,
            min_image_side: captionForm.min_image_side,
          },
        },
        start_now: false,
      });
    }
    ElMessage.success(
      stage.value === "tag" && chainCaption.value
        ? "Tag job " + (startNow ? "started" : "queued") + " + caption job queued"
        : startNow
          ? "Prep job started"
          : "Prep job queued"
    );
    await router.push("/prep");
  } catch (e) {
    formError.value = formatError(e);
  } finally {
    submitting.value = false;
  }
}

onMounted(() => {
  void loadModels();
});
</script>

<style scoped>
.page-head {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.prep-form-title {
  font-size: 16px;
  font-weight: 600;
  text-transform: capitalize;
}
.prep-form-body {
  max-width: 640px;
}
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
.form-actions {
  display: flex;
  gap: 8px;
  margin-top: 20px;
}
.model-radio-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}
.model-radio {
  height: auto;
  display: flex;
  width: 100%;
  align-items: baseline;
}
.quant-radio-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}
.hint-text {
  display: block;
  margin-top: 4px;
  margin-bottom: 8px;
}
.w-full {
  width: 100%;
}
.ml-8 {
  margin-left: 8px;
}
.mt-8 {
  margin-top: 8px;
}
.mb-12 {
  margin-bottom: 12px;
}
.mt-12 {
  margin-top: 12px;
}

@media (max-width: 600px) {
  .form-row-2 {
    grid-template-columns: 1fr;
  }
}
.preset-desc {
  margin-top: 4px;
  display: block;
}
</style>

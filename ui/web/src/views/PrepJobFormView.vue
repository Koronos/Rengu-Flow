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
              <FieldPathTag path="path" />
            </template>
            <PathFieldControl
              v-model="form.path"
              expect="dir"
              required
              placeholder="/path/to/dataset"
              input-class="w-full"
            />
          </el-form-item>

          <!-- Cleanup reads and writes images only — caption layout is irrelevant to it. -->
          <div v-if="stage !== 'clean'" class="form-row-2">
            <el-form-item>
              <template #label>
                Caption format <FieldHelpIcon :field="help('Sidecar: one .txt per image, each line a caption variant (line 1 = tags, line 2 = caption). JSON: single captions.json index file per folder.')" />
                <FieldPathTag path="caption_format" />
              </template>
              <el-select v-model="form.caption_format" class="w-full">
                <el-option label="Caption files (.txt next to each image)" value="sidecar" />
                <el-option label="captions.json (single index file)" value="json" />
              </el-select>
            </el-form-item>
            <el-form-item v-if="form.caption_format !== 'json'">
              <template #label>Caption extension <FieldPathTag path="caption_ext" /></template>
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
                Models <FieldHelpIcon :field="help('Runs each model in sequence and merges per-image probabilities by max, so the ensemble catches tags any single model misses. Add a second model when one alone keeps missing specific tag categories.')" />
                <FieldPathTag path="tag.models" />
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
                  General tag confidence <FieldHelpIcon :field="help('Sets the probability floor for all general tags across every selected model (leave blank = model default). Raise it if output is cluttered with low-confidence noise; lower it if rare or subtle tags keep disappearing.')" />
                  <FieldPathTag path="tag.general_threshold" />
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
                  Character tag confidence <FieldHelpIcon :field="help('Sets the probability floor for character and series name tags (leave blank = model default). Raise it when the tagger keeps attaching the wrong character name to images.')" />
                  <FieldPathTag path="tag.character_threshold" />
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
              <template #label><FieldPathTag path="tag.include_character_tags" /></template>
              <el-switch v-model="tagForm.include_character_tags" />
              <el-text class="ml-8" size="small">
                Include character/series name tags
                <el-text type="info"> — turn off if taggers keep mislabeling your characters, and put your own trigger word in Prepend tags instead</el-text>
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label><FieldPathTag path="tag.include_rating" /></template>
              <el-switch v-model="tagForm.include_rating" />
              <el-text class="ml-8" size="small">Include rating tag (general/sensitive/questionable/explicit)</el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Exclude tags <FieldHelpIcon :field="help('Strips these tags from every image output regardless of model confidence. Use when specific tags keep appearing that are wrong for your dataset style (e.g. realistic on anime images) or would bias training negatively.')" />
                <FieldPathTag path="tag.exclude_tags" />
              </template>
              <el-select
                v-model="tagForm.exclude_tags"
                multiple
                filterable
                allow-create
                default-first-option
                placeholder="e.g. realistic, 3d"
                class="w-full"
              >
                <el-option v-for="t in tagForm.exclude_tags" :key="t" :label="t" :value="t" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <template #label>
                Prepend tags <FieldHelpIcon :field="help('Inserts these tags at the start of every image\'s tag line before the tagger output. Use for your trigger word or any tag the model consistently misses.')" />
                <FieldPathTag path="tag.prepend_tags" />
              </template>
              <el-select
                v-model="tagForm.prepend_tags"
                multiple
                filterable
                allow-create
                default-first-option
                placeholder="e.g. my_trigger_word"
                class="w-full"
              >
                <el-option v-for="t in tagForm.prepend_tags" :key="t" :label="t" :value="t" />
              </el-select>
            </el-form-item>

            <div class="form-row-2">
              <el-form-item>
                <template #label>Max tags <FieldPathTag path="tag.max_tags" /></template>
                <el-input-number v-model="tagForm.max_tags" :min="1" :max="500" placeholder="255" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item>
                <template #label>Batch size <FieldPathTag path="tag.batch_size" /></template>
                <el-input-number v-model="tagForm.batch_size" :min="1" :max="64" placeholder="16" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <el-form-item>
              <template #label>
                Overwrite <FieldHelpIcon :field="help('Re-tags images that already have a tag line on line 1, replacing it. Turn on when you are changing models or thresholds and want to regenerate tags for the whole folder from scratch.')" />
                <FieldPathTag path="tag.overwrite" />
              </template>
              <el-switch v-model="tagForm.overwrite" />
              <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
            </el-form-item>

            <el-form-item>
              <el-switch v-model="chainCaption" />
              <el-text class="ml-8" size="small">
                Also queue a caption job immediately after this tag job
                <el-text type="info"> — use when you want tag + caption in one go; queue the caption job separately if you need custom prompt or model settings</el-text>
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
                Model <FieldHelpIcon :field="help('JoyCaption (8B, bf16 ~17 GB) writes free-form captions from a composable instruction prompt. ToriiGate (~5B) is an anime specialist that uses your tag line as grounding — pick it when caption style consistency with Danbooru vocabulary matters more than prompt flexibility.')" />
                <FieldPathTag path="caption.model" />
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

            <el-alert
              v-if="captionForm.model === 'toriigate-0.5'"
              type="info"
              :closable="false"
              show-icon
              class="mt-8 mb-12"
              title="ToriiGate uses fixed internal prompt formats: 'Concise' maps to its short format, all other bases map to long. Custom modifiers are appended as extra requirements. Open the prompt preview below to see exactly what the model receives — if the output captions read oddly, that is the place to look first."
            />

            <el-form-item>
              <template #label>
                Quantization <FieldHelpIcon :field="help('Controls how model weights are stored in VRAM. Start with bf16; drop to int8 or nf4 if the job fails with an out-of-memory error.')" />
                <FieldPathTag path="caption.quantization" />
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
                  <el-text size="small" type="info"> (smallest VRAM — use when int8 still OOMs)</el-text>
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item>
              <template #label>
                Prompt base <FieldHelpIcon :field="help('Sets the core intent of every generated caption. Change it when the default captions are too verbose, or when you need captions centered on a specific aspect (character, style). A custom prompt in the field below overrides the whole composition.')" />
                <FieldPathTag path="caption.prompt_base" />
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
                Prompt modifiers (stackable) <FieldHelpIcon :field="help('Each modifier adds one instruction to the base prompt — they compose independently, so tick only what your dataset needs. Check each modifier\'s description and the prompt preview to see the exact effect before running a large job.')" />
                <FieldPathTag path="caption.prompt_modifiers" />
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
                  Character trigger name (optional) <FieldHelpIcon :field="help('The model replaces inherent traits (hair, eye color, face) with this name so they collapse into the trigger token at training time. Set it when you want prompting the trigger to reliably reproduce the character\'s fixed appearance without describing those traits explicitly.')" />
                  <FieldPathTag path="caption.character_name" />
                </template>
                <el-input
                  v-model="captionForm.character_name"
                  placeholder="e.g. hatsune miku — inherent traits stay in the name"
                  class="w-full"
                />
                <el-text v-if="captionForm.character_name.trim()" size="small" type="info" class="hint-text">
                  A scrubber removes any trait clauses the model leaks anyway, keeping absorption consistent. Disabled when a canonical look is set.
                </el-text>
              </el-form-item>
              <el-form-item>
                <template #label>
                  Outfit policy (with trigger) <FieldHelpIcon :field="help('Controls whether the outfit is described (stays swappable at generation) or absorbed into the trigger (default outfit always appears). Mixed 50/50 gives the model both signals — use it when you want outfit swapping to work but the canonical look as the default.')" />
                  <FieldPathTag path="caption.outfit" />
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
                Canonical look (optional — for datasets with character variants) <FieldHelpIcon :field="help('Describe the character\'s baseline appearance (e.g. &quot;aqua twin-tail hair, blue eyes&quot;). Traits that match are absorbed into the trigger; deviations (aged-up versions, alternate hairstyles) are described, keeping them promptable. Use only when your dataset deliberately mixes canon and variant images.')" />
                <FieldPathTag path="caption.character_canon" />
              </template>
              <el-input
                v-model="captionForm.character_canon"
                type="textarea"
                :rows="2"
                placeholder="e.g. aqua twin-tail hair, blue eyes, slim teenage build — deviations (aged-up, alternate hairstyle, meme forms) get described instead of absorbed"
                class="w-full"
              />
              <el-text v-if="captionForm.character_canon.trim()" size="small" type="warning">
                The trait scrubber is disabled in canon mode — if the model fails to separate a variant from canon, those trait clauses will remain in the caption.
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Caption line <FieldHelpIcon :field="help('Writes the caption to this line in each sidecar file, leaving all other lines untouched. Use 3+ to add a second caption variant — each line is treated as an independent caption at training time.')" />
                <FieldPathTag path="caption.target_line" />
              </template>
              <el-input-number
                v-model="captionForm.target_line"
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
              <template #label>Custom prompt (overrides the composition) <FieldPathTag path="caption.prompt" /></template>
              <el-input
                v-model="captionForm.prompt"
                type="textarea"
                :rows="3"
                :placeholder="previewText || 'model default'"
                class="w-full"
              />
              <el-collapse class="prompt-preview-collapse mt-8">
                <el-collapse-item name="preview">
                  <template #title>
                    Prompt preview (exactly what the model receives)
                    <el-tag v-if="previewNative" size="small" type="warning" class="ml-8">native ToriiGate format</el-tag>
                  </template>
                  <pre class="prompt-preview-pre">{{ previewText || '(waiting for model selection…)' }}</pre>
                </el-collapse-item>
              </el-collapse>
            </el-form-item>

            <div class="form-row-2">
              <el-form-item>
                <template #label>Batch size <FieldPathTag path="caption.batch_size" /></template>
                <el-input-number v-model="captionForm.batch_size" :min="1" :max="16" placeholder="4" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item>
                <template #label>Max new tokens <FieldPathTag path="caption.max_new_tokens" /></template>
                <el-input-number v-model="captionForm.max_new_tokens" :min="32" :max="4096" placeholder="512" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Max image side (px, 0 = no downscale) <FieldHelpIcon :field="help('Downscales the long side of each image to this limit before the VLM sees it (default 1536 px). Lower it to reduce VRAM per image when captioning very large originals; training bucketing handles the real resize independently.')" />
                  <FieldPathTag path="caption.max_image_side" />
                </template>
                <el-input-number v-model="captionForm.max_image_side" :min="0" :step="128" placeholder="1536" controls-position="right" class="w-full" />
                <el-text v-if="captionForm.model === 'toriigate-0.5'" size="small" type="info" class="hint-text">
                  ToriiGate additionally caps inputs at ~1.0 Mpx (its training resolution).
                </el-text>
              </el-form-item>
              <el-form-item>
                <template #label>
                  Min image side (px, 0 = no filter) <FieldHelpIcon :field="help('Skips images whose short side is below this limit. Set it when your folder contains thumbnails or web-scrape artifacts that produce garbled captions.')" />
                  <FieldPathTag path="caption.min_image_side" />
                </template>
                <el-input-number v-model="captionForm.min_image_side" :min="0" :step="64" placeholder="0" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Temperature <FieldHelpIcon :field="help('Controls output randomness; leave blank for the recommended value of the selected model (shown in the placeholder). Lower it for more consistent captions across similar images; raise it only if captions feel repetitive.')" />
                  <FieldPathTag path="caption.temperature" />
                </template>
                <el-input-number
                  v-model="captionForm.temperature"
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
                  v-model="captionForm.top_p"
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

            <el-form-item v-if="captionForm.model === 'toriigate-0.5'">
              <template #label><FieldPathTag path="caption.exact_generation" /></template>
              <el-switch v-model="captionForm.exact_generation" />
              <el-text class="ml-8" size="small">
                Exact (unpadded) generation
                <el-text type="info"> — generates one image at a time instead of in batches, giving bit-exact results (~2.5x slower); turn on if batched captions for similar images are phrased inconsistently</el-text>
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label><FieldPathTag path="caption.use_tags_as_grounding" /></template>
              <el-switch v-model="captionForm.use_tags_as_grounding" />
              <el-text class="ml-8" size="small">
                Use tags as grounding
                <el-text type="info"> — feeds line-1 booru tags to ToriiGate as context, improving tag/caption consistency; turn off only if the tag line is absent or unreliable</el-text>
              </el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Overwrite <FieldHelpIcon :field="help('Re-captions images that already have content on the target line, replacing it. Turn on when you are changing the prompt or model and want to regenerate captions for the whole folder.')" />
                <FieldPathTag path="caption.overwrite" />
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
                Confidence threshold <FieldHelpIcon :field="help('YOLO11 detection score required to flag a region as a watermark (default 0.35). Lower it if the detector keeps missing faint or small watermarks; raise it if clean areas are being incorrectly inpainted.')" />
                <FieldPathTag path="clean.confidence" />
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
                Mask dilation (px) <FieldHelpIcon :field="help('Expands each detected watermark region by this many pixels before inpainting (default 8). Increase it when inpainted areas show a leftover fringe around the original watermark edge.')" />
                <FieldPathTag path="clean.mask_dilation_px" />
              </template>
              <el-input-number v-model="cleanForm.mask_dilation_px" :min="0" :max="100" placeholder="8" controls-position="right" />
            </el-form-item>

            <el-form-item>
              <template #label>
                In-place <FieldHelpIcon :field="help('Overwrites originals rather than writing to a separate folder (originals are backed up under the app data dir first, not inside the dataset). Use it when you want the training folder itself to contain only cleaned images and do not need the side-by-side comparison.')" />
                <FieldPathTag path="clean.in_place" />
              </template>
              <el-switch v-model="cleanForm.in_place" />
              <el-text class="ml-8" size="small">In-place (overwrite originals)</el-text>
              <el-alert
                v-if="cleanForm.in_place"
                type="warning"
                show-icon
                :closable="false"
                class="mt-8"
                title="Originals are backed up under the app data dir before cleaning"
              />
            </el-form-item>

            <el-form-item v-if="!cleanForm.in_place">
              <template #label>
                Output directory <FieldHelpIcon :field="help('Where to write cleaned images. Defaults to <dataset>/cleaned/.')" />
                <FieldPathTag path="clean.output_dir" />
              </template>
              <PathFieldControl
                v-model="cleanForm.output_dir"
                expect="dir"
                placeholder="<dataset>/cleaned"
                input-class="w-full"
              />
            </el-form-item>

            <el-form-item>
              <template #label><FieldPathTag path="clean.copy_undetected" /></template>
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
import { computed, onMounted, reactive, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import FieldHelpIcon from "../components/FieldHelpIcon.vue";
import FieldPathTag from "../components/FieldPathTag.vue";
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
  temperature: null as number | null,
  top_p: null as number | null,
  exact_generation: false,
  batch_size: 4,
  use_tags_as_grounding: true,
  overwrite: false,
  max_image_side: 1536,
  min_image_side: 0,
});

// --- clean form ---
const cleanForm = reactive({
  confidence: 0.35,
  mask_dilation_px: 8,
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

function samplingDefaultsPlaceholder(field: "temperature" | "top_p"): string {
  const defaults = promptOptions.value?.sampling_defaults?.[captionForm.model];
  if (!defaults) return "model default";
  const val = defaults[field];
  if (val == null) return "model default";
  return `model default (${val})`;
}

// --- server-side prompt preview ---
const previewText = ref("");
const previewNative = ref(false);
let _previewGen = 0;
let _previewTimer: ReturnType<typeof setTimeout> | null = null;

function captionConfigSnapshot() {
  return {
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
    overwrite: captionForm.overwrite,
    max_image_side: captionForm.max_image_side,
    min_image_side: captionForm.min_image_side,
  };
}

function schedulePreview() {
  if (stage.value !== "caption") return;
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
    captionForm.model,
    captionForm.prompt,
    captionForm.prompt_base,
    captionForm.prompt_modifiers.slice(),
    captionForm.character_name,
    captionForm.character_canon,
    captionForm.outfit,
    captionForm.use_tags_as_grounding,
  ],
  () => schedulePreview(),
  { deep: true }
);

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
        exact_generation: captionForm.exact_generation,
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
            exact_generation: captionForm.exact_generation,
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
.prompt-preview-collapse {
  width: 100%;
}
.prompt-preview-pre {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
  background: var(--el-fill-color-light);
  border-radius: 4px;
  padding: 8px 10px;
  margin: 0;
  max-height: 260px;
  overflow-y: auto;
  color: var(--el-text-color-secondary);
}
</style>

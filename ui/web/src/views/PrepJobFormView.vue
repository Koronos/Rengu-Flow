<template>
  <div class="prep-form-page page-shell">
    <div class="page-head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Dataset Studio</el-button>
      <span class="prep-form-title">New {{ stageLabel }} job</span>
    </div>

    <el-alert v-if="formError" type="error" :title="formError" show-icon class="mt-12" />

    <div class="prep-form-layout">
      <el-card shadow="never" class="prep-form-card">
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
              placeholder="e.g. /path/to/dataset"
              input-class="w-full"
            />
          </el-form-item>

          <!-- Cleanup and quality filter read images only — caption layout is irrelevant to them. -->
          <div v-if="stage !== 'clean' && stage !== 'quality'" class="form-row-2">
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
              <template #label>
                Caption extension <FieldHelpIcon :field="help('Extension of the per-image sidecar files read and written (default .txt). Change it only if your trainer or downstream tooling expects a different one.')" />
                <FieldPathTag path="caption_ext" />
              </template>
              <el-input v-model="form.caption_ext" placeholder=".txt" class="w-full" />
            </el-form-item>
          </div>
        </el-form>

        <el-divider class="section-divider" />

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

            <!-- Per-model confidence floors: each model keeps its own, seeded from its
                 defaults. 0 drops Character/Rating for that model; General is always kept. -->
            <h4 class="section-subtitle">
              Confidence per model <FieldHelpIcon :field="help('Each selected model has its own confidence floors, pre-filled with that model\'s own defaults. Set Character or Rating to 0 to drop that category for that model; general tags are always kept. Higher = fewer but surer tags.')" />
            </h4>
            <el-text v-if="!tagForm.models.length" size="small" type="info" class="hint-text">
              Select at least one model above to set its confidence.
            </el-text>
            <div
              v-for="mid in tagForm.models"
              v-else
              :key="mid"
              class="model-thresholds"
            >
              <span class="model-thresholds__name">{{ mid }}</span>
              <div class="model-thresholds__fields">
                <div class="conf-field">
                  <label class="conf-label">General</label>
                  <el-input-number
                    v-model="tagThresholds[mid].general"
                    :min="0"
                    :max="1"
                    :step="0.05"
                    :precision="2"
                    controls-position="right"
                    class="conf-input"
                  />
                </div>
                <div class="conf-field">
                  <label class="conf-label">Character <span class="conf-off">0 = off</span></label>
                  <el-input-number
                    v-model="tagThresholds[mid].character"
                    :min="0"
                    :max="1"
                    :step="0.05"
                    :precision="2"
                    controls-position="right"
                    class="conf-input"
                  />
                </div>
                <div class="conf-field">
                  <label class="conf-label">Rating <span class="conf-off">0 = off</span></label>
                  <el-input-number
                    v-model="tagThresholds[mid].rating"
                    :min="0"
                    :max="1"
                    :step="0.05"
                    :precision="2"
                    controls-position="right"
                    class="conf-input"
                  />
                </div>
              </div>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Exclude tags <FieldHelpIcon :field="help('Strips these tags from every image output regardless of model confidence. Use when specific tags keep appearing that are wrong for your dataset style (e.g. realistic on anime images) or would bias training negatively.')" />
                  <FieldPathTag path="tag.exclude_tags" />
                </template>
                <el-input-tag
                  v-model="tagForm.exclude_tags"
                  clearable
                  delimiter=","
                  placeholder="e.g. realistic, 3d"
                  class="w-full"
                />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Prepend tags <FieldHelpIcon :field="help('Inserts these tags at the start of every image\'s tag line before the tagger output. Use for your trigger word or any tag the model consistently misses.')" />
                  <FieldPathTag path="tag.prepend_tags" />
                </template>
                <el-input-tag
                  v-model="tagForm.prepend_tags"
                  clearable
                  delimiter=","
                  placeholder="e.g. my_trigger_word"
                  class="w-full"
                />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Max tags <FieldHelpIcon :field="help('Hard cap on tags kept per image, highest-confidence first (default 40). Raise it if useful tags are being dropped; lower it to trim the low-confidence tail.')" />
                  <FieldPathTag path="tag.max_tags" />
                </template>
                <el-input-number v-model="tagForm.max_tags" :min="1" :max="500" placeholder="40" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Batch size <FieldHelpIcon :field="help('Images per ONNX forward pass (default 8). Raise it to tag faster on a card with spare VRAM; lower it if the job runs out of memory.')" />
                  <FieldPathTag path="tag.batch_size" />
                </template>
                <el-input-number v-model="tagForm.batch_size" :min="1" :max="64" placeholder="8" controls-position="right" class="w-full" />
              </el-form-item>
            </div>

            <div class="form-row-2">
              <el-form-item>
                <template #label>
                  Overwrite <FieldHelpIcon :field="help('Re-tags images that already have a tag line on line 1, replacing it. Turn on when you are changing models or thresholds and want to regenerate tags for the whole folder from scratch.')" />
                  <FieldPathTag path="tag.overwrite" />
                </template>
                <el-switch v-model="tagForm.overwrite" />
                <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
              </el-form-item>
              <el-form-item>
                <template #label>
                  Quality tags <FieldHelpIcon :field="help('Runs the deepghs aesthetic model and prepends a booru quality tag (masterpiece … worst quality) to each caption, the anime-training convention. Adds a GPU pass and downloads the model on first use.')" />
                  <FieldPathTag path="tag.quality_tags" />
                </template>
                <el-switch v-model="tagForm.quality_tags" />
                <el-text class="ml-8" size="small">Prepend quality tag to each caption</el-text>
              </el-form-item>
              <el-form-item>
                <template #label>
                  Underscores <FieldHelpIcon :field="help('Tag form: on keeps the original danbooru form (long_hair), off writes spaces (long hair). SDXL is usually trained with underscores; Cosmos is spaces-only. Tag-dropout control lists (undroppable tags) match both forms either way, so the choice only affects the written captions.')" />
                  <FieldPathTag path="tag.underscores" />
                </template>
                <el-switch v-model="tagForm.underscores" />
                <el-text class="ml-8" size="small">Keep original danbooru form (long_hair)</el-text>
              </el-form-item>
              <el-form-item>
                <template #label>
                  Chain a caption job <FieldHelpIcon :field="help('Queues a caption job on the same folder right after this tag job, so tagging then captioning run back-to-back. Leave off and queue the caption job separately when you need custom prompt or model settings.')" />
                </template>
                <el-switch v-model="chainCaption" />
                <el-text class="ml-8" size="small">Also queue a caption job immediately after this tag job</el-text>
              </el-form-item>
            </div>
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
                  Character trigger name <FieldHelpIcon :field="help('The model replaces inherent traits (hair, eye color, face) with this name so they collapse into the trigger token at training time. Set it when you want prompting the trigger to reliably reproduce the character\'s fixed appearance without describing those traits explicitly.')" />
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
                Canonical look <FieldHelpIcon :field="help('For datasets with character variants: describe the character\'s baseline appearance (e.g. &quot;aqua twin-tail hair, blue eyes&quot;). Traits that match are absorbed into the trigger; deviations (aged-up versions, alternate hairstyles) are described, keeping them promptable. Use only when your dataset deliberately mixes canon and variant images.')" />
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
                <el-input-number v-model="captionForm.batch_size" :min="1" :max="16" placeholder="4" controls-position="right" class="w-full" />
              </el-form-item>
              <el-form-item>
                <template #label>
                  Max new tokens <FieldHelpIcon :field="help('Upper bound on caption length in tokens (default 512). Raise it if long captions get cut off mid-sentence; lower it to keep captions terse and save time.')" />
                  <FieldPathTag path="caption.max_new_tokens" />
                </template>
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
              <template #label>
                Exact generation <FieldHelpIcon :field="help('Generates one image at a time instead of in batches, giving bit-exact results (~2.5x slower). Turn on if batched captions for similar images come out phrased inconsistently.')" />
                <FieldPathTag path="caption.exact_generation" />
              </template>
              <el-switch v-model="captionForm.exact_generation" />
              <el-text class="ml-8" size="small">Exact (unpadded) generation</el-text>
            </el-form-item>

            <el-form-item>
              <template #label>
                Use tags as grounding <FieldHelpIcon :field="help('Feeds the line-1 booru tags to ToriiGate as context, improving tag/caption consistency. Turn off only if the tag line is absent or unreliable.')" />
                <FieldPathTag path="caption.use_tags_as_grounding" />
              </template>
              <el-switch v-model="captionForm.use_tags_as_grounding" />
              <el-text class="ml-8" size="small">Use tags as grounding</el-text>
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
              <template #label>
                Copy undetected images <FieldHelpIcon :field="help('Also copies images with no watermark detected into the output folder, so it ends up as a complete cleaned dataset. Turn off to write only the images that were actually inpainted.')" />
                <FieldPathTag path="clean.copy_undetected" />
              </template>
              <el-switch v-model="cleanForm.copy_undetected" />
              <el-text class="ml-8" size="small">Copy images with no detections to output</el-text>
            </el-form-item>
          </el-form>
        </template>

        <!-- Quality stage -->
        <template v-if="stage === 'quality'">
          <h3 class="section-title">Quality filter options</h3>
          <el-alert
            type="info"
            :closable="false"
            show-icon
            class="mt-8 mb-12"
            title="Scans images for blur or low aesthetic quality and flags (or moves) them. Non-destructive by default — report mode only lists flagged images without moving them."
          />
          <el-form label-position="top">
            <el-form-item>
              <template #label>
                Metric <FieldHelpIcon :field="help('Blur: fast Laplacian + resolution heuristic, no download. Aesthetic: anime booru appeal model (worst→masterpiece), downloads on first use. Technical IQA: learned No-Reference model scoring perceived technical quality (blur, noise, compression) — works on anime and natural photos, downloads a model on first use.')" />
                <FieldPathTag path="quality.metric" />
              </template>
              <el-select v-model="qualityForm.metric" class="w-full">
                <el-option label="Blur / resolution (fast, no download)" value="blur" />
                <el-option label="Aesthetic — booru quality (deepghs, downloads a model)" value="aesthetic" />
                <el-option label="Technical IQA — image quality (pyiqa, anime + photo, downloads a model)" value="iqa" />
              </el-select>
            </el-form-item>

            <template v-if="qualityForm.metric === 'blur'">
              <el-form-item>
                <template #label>
                  Blur threshold <FieldHelpIcon :field="help('Laplacian-variance floor measured on a long-side-512 copy; images below it are flagged blurry (default 80). Run in report mode first, then set the threshold between your good and bad samples.')" />
                  <FieldPathTag path="quality.blur_threshold" />
                </template>
                <el-input-number
                  v-model="qualityForm.blur_threshold"
                  :min="0"
                  placeholder="80"
                  controls-position="right"
                />
              </el-form-item>

              <el-form-item>
                <template #label>
                  Min side (px) <FieldHelpIcon :field="help('Flag images whose shorter side (pixels) is below this value (default 0 = off). Use it to catch undersized images that would degrade training resolution buckets.')" />
                  <FieldPathTag path="quality.min_side" />
                </template>
                <el-input-number
                  v-model="qualityForm.min_side"
                  :min="0"
                  placeholder="0"
                  controls-position="right"
                />
              </el-form-item>

              <el-form-item>
                <template #label>
                  Min detail <FieldHelpIcon :field="help('Effective-resolution floor: flags pixelated or upscaled images (a 64px picture blown up to 1024 still reads as sharp to the blur check, but has little real detail). 0 disables it. Typical starting point ~12–15; calibrate in report mode.')" />
                  <FieldPathTag path="quality.min_detail" />
                </template>
                <el-input-number
                  v-model="qualityForm.min_detail"
                  :min="0"
                  :step="0.5"
                  placeholder="0"
                  controls-position="right"
                />
              </el-form-item>
            </template>

            <el-form-item v-if="qualityForm.metric === 'aesthetic'">
              <template #label>
                Minimum label <FieldHelpIcon :field="help('Flags any image the booru-quality model ranks below this tier (default normal). Higher = stricter; moving up to good will flag the worst, low, and normal tiers.')" />
                <FieldPathTag path="quality.aesthetic_min_label" />
              </template>
              <el-select v-model="qualityForm.aesthetic_min_label" class="w-full">
                <el-option label="worst — flag nothing" value="worst" />
                <el-option label="low — flag worst" value="low" />
                <el-option label="normal — flag worst &amp; low" value="normal" />
                <el-option label="good — flag worst, low &amp; normal" value="good" />
                <el-option label="great — flag everything below great" value="great" />
                <el-option label="best — flag everything below best" value="best" />
                <el-option label="masterpiece — flag everything below masterpiece" value="masterpiece" />
              </el-select>
            </el-form-item>

            <template v-if="qualityForm.metric === 'iqa'">
              <el-form-item>
                <template #label>
                  Model <FieldHelpIcon :field="help('Which NR-IQA model scores quality. clipiqa/arniqa generalize to illustration; musiq/maniqa are tuned on natural photos; brisque/niqe are classic baselines.')" />
                  <FieldPathTag path="quality.iqa_model" />
                </template>
                <el-select v-model="qualityForm.iqa_model" class="w-full">
                  <el-option label="CLIP-IQA — any domain (anime + photo)" value="clipiqa" />
                  <el-option label="ARNIQA — any domain, robust" value="arniqa" />
                  <el-option label="MUSIQ — natural photos" value="musiq" />
                  <el-option label="MANIQA — natural photos" value="maniqa" />
                  <el-option label="BRISQUE — classic, photos" value="brisque" />
                  <el-option label="NIQE — classic, opinion-free" value="niqe" />
                </el-select>
              </el-form-item>

              <el-form-item>
                <template #label>
                  Discard lowest % <FieldHelpIcon :field="help('Flags the lowest-quality N% of the dataset, ranked by the selected model (same behavior for every model). 10 = drop the worst 10%; raise it to cull more. Run report mode first to see how many fall at each level.')" />
                  <FieldPathTag path="quality.iqa_threshold" />
                </template>
                <el-slider
                  v-model="qualityForm.iqa_threshold"
                  :min="0"
                  :max="100"
                  :step="1"
                  show-input
                  class="w-full"
                />
              </el-form-item>
            </template>

            <el-form-item>
              <template #label>
                Move flagged <FieldHelpIcon :field="help('Moves flagged images into &lt;path&gt;/low_quality/ (off = report only, non-destructive). Use report mode first to review what would be flagged before enabling move.')" />
                <FieldPathTag path="quality.action" />
              </template>
              <el-switch v-model="qualityForm.move" />
              <el-text class="ml-8" size="small">Move flagged images into &lt;path&gt;/low_quality (off = report only)</el-text>
            </el-form-item>

            <el-form-item v-if="qualityForm.move">
              <template #label>
                Output directory <FieldHelpIcon :field="help('Where to move flagged images. Defaults to &lt;path&gt;/low_quality.')" />
                <FieldPathTag path="quality.output_dir" />
              </template>
              <PathFieldControl
                v-model="qualityForm.output_dir"
                expect="dir"
                placeholder="e.g. /data/rejects (default: <path>/low_quality)"
                input-class="w-full"
              />
            </el-form-item>
          </el-form>
        </template>

        <!-- Submit actions -->
        <div class="form-actions">
          <el-button @click="$router.push('/prep')">Cancel</el-button>
          <el-button :loading="submitting" @click="submit(false)">Queue</el-button>
          <el-button v-if="canPreview" type="primary" :loading="previewRunning" @click="runPreview">Preview report</el-button>
          <el-button v-else type="primary" :loading="submitting" @click="submit(true)">Start now</el-button>
        </div>

        <!-- Inline quality report preview -->
        <div v-if="previewJobId" class="preview-report-panel mt-8">
          <div v-if="previewRunning" class="preview-progress">
            <el-text size="small" class="hint-text">{{ previewProgress?.msg || 'Running…' }}</el-text>
            <el-progress :percentage="Math.min(100, Math.round(previewProgress?.percent ?? 0))" :show-text="false" class="mt-8" />
          </div>
          <el-alert v-if="previewError" type="error" :title="previewError" show-icon :closable="false" class="mt-8" />
          <template v-if="previewReport">
            <h3 class="section-title mt-8">Report (nothing was changed — report only)</h3>
            <el-descriptions :column="3" border size="small">
              <el-descriptions-item
                v-for="(val, key) in previewReportFields"
                :key="key"
                :label="String(key)"
              >{{ val }}</el-descriptions-item>
            </el-descriptions>
            <el-text
              v-if="Number(previewReport.flagged) > 0"
              size="small"
              class="hint-text preview-flagged mt-8"
            >
              {{ previewReport.flagged }} of {{ previewReport.scored }} images would be flagged
            </el-text>
            <el-text
              v-if="previewReasonsSummary"
              size="small"
              class="hint-text preview-reasons mt-4"
            >
              {{ previewReasonsSummary }}
            </el-text>
          </template>
        </div>
      </div>
      </el-card>

      <aside class="prep-summary">
        <PrepJobSummaryPanel
          :stage="stage"
          :form="form"
          :tag-form="tagForm"
          :tag-thresholds="tagThresholds"
          :caption-form="captionForm"
          :clean-form="cleanForm"
          :quality-form="qualityForm"
          :prompt-options="promptOptions"
          :preview-text="previewText"
          :preview-native="previewNative"
        />
      </aside>
    </div>
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
import PrepJobSummaryPanel from "../components/PrepJobSummaryPanel.vue";
import { formatError } from "../lib/formatError";
import { usePrepJobLive } from "../composables/usePrepJobLive";
import type { PrepModelInfo, PrepPromptOptions, PrepStage } from "../types/api";

function help(text: string) {
  return { path: "", type: "string", help: text, doc_path: "docs/user/dataset-prep.md" };
}

const route = useRoute();
const router = useRouter();

const stage = computed(() => (route.params.stage as PrepStage) || "tag");
const stageLabel = computed(() => {
  const map: Record<string, string> = { tag: "tag", caption: "caption", clean: "clean", quality: "quality" };
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
  quality_tags: false,
  underscores: false,
});

// Per-model confidence floors (general/character/rating), always populated from each
// model's own defaults. 0 = drop that category for that model (general always kept).
interface ModelThresholds {
  general: number;
  character: number;
  rating: number;
}
const tagThresholds = reactive<Record<string, ModelThresholds>>({});

function modelDefaults(modelId: string): ModelThresholds {
  const m = tagModels.value.find((x) => x.id === modelId);
  return {
    general: m?.general_threshold ?? 0.35,
    character: m?.character_threshold ?? 0.85,
    rating: m?.rating_threshold ?? 0.5,
  };
}

/** Ensure every selected model has a thresholds row, seeded from its defaults. */
function syncThresholds(): void {
  for (const id of tagForm.models) {
    if (!tagThresholds[id]) tagThresholds[id] = modelDefaults(id);
  }
}

watch(() => tagForm.models.slice(), syncThresholds, { deep: true });

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

// --- quality form ---
const qualityForm = reactive({
  metric: "blur" as "blur" | "aesthetic" | "iqa",
  blur_threshold: 80,
  min_side: 0,
  min_detail: 0,
  aesthetic_min_label: "normal",
  iqa_model: "clipiqa",
  iqa_threshold: 10,
  move: false,
  output_dir: "",
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
    captionForm.prompt = val;
  } else {
    promptDirty.value = false;
    captionForm.prompt = "";
  }
}

function resetPromptToComposed(): void {
  promptDirty.value = false;
  captionForm.prompt = "";
  promptText.value = composedText.value;
}

// --- quality inline preview (report-only) ---
const canPreview = computed(() => stage.value === "quality" && !qualityForm.move);
const previewJobId = ref<string | null>(null);
const previewReport = ref<Record<string, unknown> | null>(null);
const previewRunning = ref(false);
const previewError = ref("");

const { progress: previewProgress } = usePrepJobLive(
  () => previewJobId.value ?? undefined,
  { onRunFinished: onPreviewFinished }
);

const PREVIEW_EXCLUDED = new Set(["failed", "errors", "low_quality"]);
const previewReportFields = computed(() => {
  if (!previewReport.value) return {} as Record<string, unknown>;
  return Object.fromEntries(
    Object.entries(previewReport.value).filter(([k]) => !PREVIEW_EXCLUDED.has(k))
  ) as Record<string, unknown>;
});

const previewReasonsSummary = computed(() => {
  const lq = previewReport.value?.low_quality;
  if (!Array.isArray(lq) || lq.length === 0) return "";
  const counts: Record<string, number> = {};
  for (const item of lq) {
    const reasons = (item as Record<string, unknown>).reasons;
    if (Array.isArray(reasons)) {
      for (const r of reasons) {
        const s = String(r);
        counts[s] = (counts[s] ?? 0) + 1;
      }
    }
  }
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([r, n]) => `${r}: ${n}`)
    .join(" · ");
});

async function onPreviewFinished(): Promise<void> {
  const id = previewJobId.value;
  if (!id) return;
  try {
    const res = await api.prepJobReport(id);
    previewReport.value = res.report;
  } catch {
    // report.json may lag the finish signal
    await new Promise<void>((r) => setTimeout(r, 600));
    try {
      const res = await api.prepJobReport(id);
      previewReport.value = res.report;
    } catch (e) {
      previewError.value = formatError(e);
    }
  }
  previewRunning.value = false;
}

async function runPreview(): Promise<void> {
  previewError.value = "";
  previewReport.value = null;
  if (!form.path.trim()) {
    previewError.value = "Dataset folder is required.";
    return;
  }
  previewRunning.value = true;
  try {
    const job = await api.createPrepJob({ stage: stage.value, config: buildConfig(), start_now: true });
    previewJobId.value = String(job.id);
  } catch (e) {
    previewError.value = formatError(e);
    previewRunning.value = false;
  }
}

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
      syncThresholds();
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
    // Each model carries its own confidence floors as a per-model override; a 0 floor
    // for character/rating turns that category off for that model (general is kept).
    const overrides: Record<string, Record<string, number | boolean>> = {};
    for (const id of tagForm.models) {
      const t = tagThresholds[id] ?? modelDefaults(id);
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
        overwrite: tagForm.overwrite,
        quality_tags: tagForm.quality_tags,
        underscores: tagForm.underscores,
        overrides,
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
  if (stage.value === "quality") {
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

/** Copy only the keys the target form already knows; unknown/stale keys are dropped. */
function copyKnown(target: Record<string, unknown>, src: unknown): void {
  if (!src || typeof src !== "object") return;
  const s = src as Record<string, unknown>;
  for (const key of Object.keys(target)) {
    if (s[key] !== undefined && s[key] !== null) target[key] = s[key];
  }
}

/** Seed the form from an existing job's parsed config (clone "new job from this"). */
function applyConfig(cfg: Record<string, unknown>): void {
  if (!cfg || typeof cfg !== "object") return;
  if (typeof cfg.path === "string") form.path = cfg.path;
  if (cfg.caption_format === "sidecar" || cfg.caption_format === "json") {
    form.caption_format = cfg.caption_format;
  }
  if (typeof cfg.caption_ext === "string") form.caption_ext = cfg.caption_ext;
  copyKnown(tagForm as unknown as Record<string, unknown>, cfg.tag);
  copyKnown(captionForm as unknown as Record<string, unknown>, cfg.caption);
  copyKnown(cleanForm as unknown as Record<string, unknown>, cfg.clean);
  copyKnown(qualityForm as unknown as Record<string, unknown>, cfg.quality);
  // action -> move conversion (copyKnown skips "action" since the form field is "move")
  const q = cfg.quality as { action?: string } | undefined;
  if (q?.action !== undefined) qualityForm.move = q.action === "move";

  // Rebuild per-model thresholds from the cloned per-model overrides (inverse of
  // buildConfig): include_*=false -> 0 (off); otherwise the stored *_threshold.
  if (stage.value === "tag") {
    syncThresholds();
    const tag = cfg.tag as { overrides?: Record<string, Record<string, unknown>> } | undefined;
    for (const [mid, o] of Object.entries(tag?.overrides ?? {})) {
      const t = tagThresholds[mid] ?? modelDefaults(mid);
      if (typeof o.general_threshold === "number") t.general = o.general_threshold;
      t.character =
        o.include_character === false
          ? 0
          : typeof o.character_threshold === "number"
            ? o.character_threshold
            : t.character;
      t.rating =
        o.include_rating === false
          ? 0
          : typeof o.rating_threshold === "number"
            ? o.rating_threshold
            : t.rating;
      tagThresholds[mid] = t;
    }
  }

  // A cloned non-empty caption prompt is a custom override, not the composition.
  if (stage.value === "caption" && captionForm.prompt.trim()) {
    promptDirty.value = true;
    promptText.value = captionForm.prompt;
  }
}

onMounted(async () => {
  await loadModels();
  const fromId = route.query.from;
  if (typeof fromId === "string" && fromId) {
    try {
      const { config } = await api.prepJobConfig(fromId);
      applyConfig(config);
    } catch {
      // best-effort — fall back to the default form
    }
  }
});
</script>

<style scoped>
.page-head {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: var(--rf-space-sm);
  flex-wrap: wrap;
}
.prep-form-title {
  font-size: 16px;
  font-weight: 600;
  text-transform: capitalize;
}
.prep-form-layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: var(--rf-space-md);
  align-items: start;
  margin-top: var(--rf-space-sm);
}
.prep-summary {
  position: sticky;
  top: var(--rf-space-md);
}
@media (max-width: 1100px) {
  .prep-form-layout {
    grid-template-columns: 1fr;
  }
  .prep-summary {
    position: static;
  }
}
.prep-form-body {
  max-width: 100%;
}
/* Keep numeric inputs compact so they don't stretch across a wide column. */
.prep-form-body :deep(.el-input-number) {
  max-width: 280px;
}
.section-title {
  margin: 0 0 12px;
  font-size: 14px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.section-divider {
  margin: var(--rf-space-md) 0;
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
.preview-report-panel {
  border-top: 1px solid var(--el-border-color-lighter);
  padding-top: 16px;
}
.preview-progress {
  padding: 4px 0;
}
.preview-flagged,
.preview-reasons {
  display: block;
}
.mt-4 {
  margin-top: 4px;
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
.prompt-state {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}
.section-subtitle {
  margin: 0 0 var(--rf-space-xs);
  font-size: 13px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
}
.model-thresholds {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--rf-space-sm) var(--rf-space-md);
  padding: var(--rf-space-sm) var(--rf-space-md);
  margin-bottom: var(--rf-space-xs);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  background: var(--el-fill-color-blank);
}
.model-thresholds__name {
  font-family: var(--rf-font-mono);
  font-size: 13px;
  font-weight: 600;
  flex: 0 0 auto;
  min-width: 140px;
}
.model-thresholds__fields {
  display: flex;
  flex-wrap: wrap;
  gap: var(--rf-space-md);
}
.conf-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.conf-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.conf-off {
  color: var(--el-text-color-placeholder);
}
.conf-input {
  width: 130px;
}
</style>

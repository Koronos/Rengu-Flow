<template>
  <div class="prep-form-page page-shell">
    <div class="page-head">
      <el-button :icon="ArrowLeft" @click="$router.push('/prep')">Prep jobs</el-button>
      <span class="prep-form-title">New {{ stageLabel }} job</span>
    </div>

    <el-alert v-if="formError" type="error" :title="formError" show-icon class="mt-12" />

    <el-card shadow="never" class="mt-12">
      <div class="prep-form-body">
        <!-- Common fields -->
        <el-form label-position="top">
          <el-form-item label="Dataset folder" required>
            <PathFieldControl
              v-model="form.path"
              expect="dir"
              required
              placeholder="/path/to/dataset"
              input-class="w-full"
            />
          </el-form-item>

          <div class="form-row-2">
            <el-form-item label="Caption format">
              <el-select v-model="form.caption_format" class="w-full">
                <el-option label="Sidecar (.txt beside image)" value="sidecar" />
                <el-option label="JSON index file" value="json" />
              </el-select>
            </el-form-item>
            <el-form-item label="Caption extension">
              <el-input v-model="form.caption_ext" placeholder=".txt" class="w-full" />
            </el-form-item>
          </div>
        </el-form>

        <el-divider />

        <!-- Tag stage -->
        <template v-if="stage === 'tag'">
          <h3 class="section-title">Tagging options</h3>
          <el-form label-position="top">
            <el-form-item label="Models">
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

            <el-form-item label="Exclude tags">
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

            <el-form-item label="Prepend tags">
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
            <el-form-item label="Model">
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
                  <el-text v-if="m.notes" size="small" type="info" class="ml-8">{{ m.notes }}</el-text>
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item label="Quantization">
              <el-radio-group v-model="captionForm.quantization">
                <el-radio value="bf16">
                  bf16
                  <el-text size="small" type="info"> (~17 GB, recommended on 24 GB)</el-text>
                </el-radio>
                <el-radio value="int8">
                  int8
                  <el-text size="small" type="info"> (smaller VRAM)</el-text>
                </el-radio>
                <el-radio value="nf4">
                  nf4
                  <el-text size="small" type="info"> (smallest VRAM)</el-text>
                </el-radio>
              </el-radio-group>
            </el-form-item>

            <el-form-item label="Prompt">
              <el-input
                v-model="captionForm.prompt"
                type="textarea"
                :rows="3"
                placeholder="model default"
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
              <el-switch v-model="captionForm.overwrite" />
              <el-text class="ml-8" size="small">Overwrite existing captions</el-text>
            </el-form-item>
          </el-form>
        </template>

        <!-- Clean stage -->
        <template v-if="stage === 'clean'">
          <h3 class="section-title">Cleaning options</h3>
          <el-form label-position="top">
            <el-form-item label="Confidence threshold">
              <el-slider
                v-model="cleanForm.confidence"
                :min="0"
                :max="1"
                :step="0.01"
                show-input
                :show-input-controls="false"
              />
            </el-form-item>

            <el-form-item label="Mask dilation (px)">
              <el-input-number v-model="cleanForm.mask_dilation_px" :min="0" :max="100" controls-position="right" />
            </el-form-item>

            <el-form-item>
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

            <el-form-item v-if="!cleanForm.in_place" label="Output directory">
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
import PathFieldControl from "../components/PathFieldControl.vue";
import { formatError } from "../lib/formatError";
import type { PrepModelInfo, PrepStage } from "../types/api";

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
});

// --- caption form ---
const captionForm = reactive({
  model: "",
  quantization: "bf16" as "bf16" | "int8" | "nf4",
  prompt: "",
  max_new_tokens: 512,
  temperature: 0.7,
  top_p: 0.9,
  batch_size: 1,
  use_tags_as_grounding: false,
  overwrite: false,
});

// --- clean form ---
const cleanForm = reactive({
  confidence: 0.3,
  mask_dilation_px: 4,
  in_place: false,
  output_dir: "",
  copy_undetected: true,
});

// --- models ---
const tagModels = ref<PrepModelInfo[]>([]);
const captionModels = ref<PrepModelInfo[]>([]);
const modelsLoading = ref(false);

async function loadModels(): Promise<void> {
  modelsLoading.value = true;
  try {
    if (stage.value === "tag") {
      const res = await api.prepModels("tag");
      tagModels.value = res.models || [];
      // pre-select downloaded models
      tagForm.models = tagModels.value.filter((m) => m.downloaded).map((m) => m.id);
    } else if (stage.value === "caption") {
      const res = await api.prepModels("caption");
      captionModels.value = res.models || [];
      const first = captionModels.value[0];
      if (first) captionForm.model = first.id;
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
        max_new_tokens: captionForm.max_new_tokens,
        temperature: captionForm.temperature,
        top_p: captionForm.top_p,
        batch_size: captionForm.batch_size,
        use_tags_as_grounding: captionForm.use_tags_as_grounding,
        overwrite: captionForm.overwrite,
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
            max_new_tokens: captionForm.max_new_tokens,
            temperature: captionForm.temperature,
            top_p: captionForm.top_p,
            batch_size: captionForm.batch_size,
            use_tags_as_grounding: captionForm.use_tags_as_grounding,
            overwrite: captionForm.overwrite,
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
  gap: 8px;
}
.model-radio {
  height: auto;
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
.mt-12 {
  margin-top: 12px;
}

@media (max-width: 600px) {
  .form-row-2 {
    grid-template-columns: 1fr;
  }
}
</style>

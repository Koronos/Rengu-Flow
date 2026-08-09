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
        <PrepCommonFields v-model="form" :stage="stage" />

        <el-divider class="section-divider" />

        <TagStageForm
          v-if="stage === 'tag'"
          v-model="tagForm"
          v-model:thresholds="tagThresholds"
          :seed="(seedConfig?.tag as PrepTagConfig) ?? null"
          @models-loaded="tagModels = $event"
        >
          <template #extra>
            <el-form-item>
              <template #label>
                Chain a caption job <FieldHelpIcon :field="help('Queues a caption job on the same folder right after this tag job, so tagging then captioning run back-to-back. Leave off and queue the caption job separately when you need custom prompt or model settings.')" />
              </template>
              <el-switch v-model="chainCaption" />
              <el-text class="ml-8" size="small">Also queue a caption job immediately after this tag job</el-text>
            </el-form-item>
          </template>
        </TagStageForm>

        <CaptionStageForm
          v-if="stage === 'caption'"
          v-model="captionForm"
          v-model:prompt-options="promptOptions"
          v-model:preview-text="previewText"
          v-model:preview-native="previewNative"
          :seed="(seedConfig?.caption as PrepCaptionConfig) ?? null"
        />

        <CleanStageForm
          v-if="stage === 'clean'"
          v-model="cleanForm"
          :seed="(seedConfig?.clean as PrepCleanConfig) ?? null"
        />

        <QualityStageForm
          v-if="stage === 'quality'"
          ref="qualityFormRef"
          v-model="qualityForm"
          v-model:preview-running="previewRunning"
          :common-form="form"
          :seed="(seedConfig?.quality as PrepQualityConfig) ?? null"
        />

        <IndexStageForm
          v-if="stage === 'index'"
          v-model="indexForm"
          :seed="(seedConfig?.index as PrepIndexConfig) ?? null"
        />

        <!-- Submit actions -->
        <div class="form-actions">
          <el-button @click="$router.push('/prep')">Cancel</el-button>
          <el-button :loading="submitting" @click="submit(false)">Queue</el-button>
          <el-button v-if="canPreview" type="primary" :loading="previewRunning" @click="qualityFormRef?.runPreview()">Preview report</el-button>
          <el-button v-else type="primary" :loading="submitting" @click="submit(true)">Start now</el-button>
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
import { computed, onMounted, reactive, ref, useTemplateRef } from "vue";
import { useRoute, useRouter } from "vue-router";
import { ElMessage } from "element-plus";
import { ArrowLeft } from "@element-plus/icons-vue";
import { api } from "../api";
import FieldHelpIcon from "../components/FieldHelpIcon.vue";
import PrepJobSummaryPanel from "../components/PrepJobSummaryPanel.vue";
import PrepCommonFields from "../components/prep/PrepCommonFields.vue";
import TagStageForm from "../components/prep/TagStageForm.vue";
import CaptionStageForm from "../components/prep/CaptionStageForm.vue";
import CleanStageForm from "../components/prep/CleanStageForm.vue";
import QualityStageForm from "../components/prep/QualityStageForm.vue";
import IndexStageForm from "../components/prep/IndexStageForm.vue";
import { help } from "../components/prep/formHelpers";
import { formatError } from "../lib/formatError";
import {
  buildStageConfig,
  defaultCaptionForm,
  defaultCleanForm,
  defaultCommonForm,
  defaultIndexForm,
  defaultQualityForm,
  defaultTagForm,
} from "../lib/prepStageConfig";
import type { ModelThresholds } from "../lib/prepStageConfig";
import type {
  PrepCaptionConfig,
  PrepCleanConfig,
  PrepIndexConfig,
  PrepModelInfo,
  PrepPromptOptions,
  PrepQualityConfig,
  PrepStage,
  PrepTagConfig,
} from "../types/api";

const route = useRoute();
const router = useRouter();

const stage = computed(() => (route.params.stage as PrepStage) || "tag");
const stageLabel = computed(() => {
  const map: Record<string, string> = {
    tag: "tag",
    caption: "caption",
    clean: "clean",
    quality: "quality",
    index: "quality index",
  };
  return map[stage.value] ?? stage.value;
});

// --- form state (owned here, edited by the stage components via v-model) ---
const form = reactive(defaultCommonForm());
const tagForm = reactive(defaultTagForm());
const tagThresholds = reactive<Record<string, ModelThresholds>>({});
const captionForm = reactive(defaultCaptionForm());
const cleanForm = reactive(defaultCleanForm());
const qualityForm = reactive(defaultQualityForm());
const indexForm = reactive(defaultIndexForm());

/** Tagger registry, reported by TagStageForm; only used to resolve defaults. */
const tagModels = ref<PrepModelInfo[]>([]);

// --- state the summary panel renders, produced by CaptionStageForm ---
const promptOptions = ref<PrepPromptOptions | null>(null);
const previewText = ref("");
const previewNative = ref(false);

// --- quality preview run (owned by QualityStageForm, triggered from here) ---
const qualityFormRef = useTemplateRef<InstanceType<typeof QualityStageForm>>("qualityFormRef");
const previewRunning = ref(false);
const canPreview = computed(() => stage.value === "quality" && !qualityForm.move);

// --- submit ---
const submitting = ref(false);
const chainCaption = ref(false);
const formError = ref("");

function buildConfig() {
  return buildStageConfig(stage.value, {
    form,
    tagForm,
    tagThresholds,
    tagModels: tagModels.value,
    captionForm,
    cleanForm,
    qualityForm,
    indexForm,
  });
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
  if (stage.value === "index" && !indexForm.models.length) {
    formError.value = "Select at least one quality model.";
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
            engine: captionForm.engine,
            vllm_quantization: captionForm.vllm_quantization,
            vllm_model: captionForm.vllm_model,
            gguf_quantization: captionForm.gguf_quantization,
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

/**
 * Config of the job being cloned ("new job from this"). The common fields are
 * applied here; each stage form seeds itself from its own section once its
 * registry/catalogue has loaded, so the clone always wins over the defaults.
 */
const seedConfig = ref<Record<string, unknown> | null>(null);

onMounted(async () => {
  const fromId = route.query.from;
  if (typeof fromId !== "string" || !fromId) return;
  try {
    const { config } = await api.prepJobConfig(fromId);
    if (!config || typeof config !== "object") return;
    if (typeof config.path === "string") form.path = config.path;
    if (config.caption_format === "sidecar" || config.caption_format === "json") {
      form.caption_format = config.caption_format;
    }
    if (typeof config.caption_ext === "string") form.caption_ext = config.caption_ext;
    seedConfig.value = config;
  } catch {
    // best-effort — fall back to the default form
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
.section-divider {
  margin: var(--rf-space-md) 0;
}
.form-actions {
  display: flex;
  gap: 8px;
  margin-top: 20px;
}
.ml-8 {
  margin-left: 8px;
}
.mt-12 {
  margin-top: 12px;
}
</style>

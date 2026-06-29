<template>
  <el-card shadow="never" class="prep-summary-card">
    <template #header>
      <span>Summary</span>
    </template>

    <dl class="prep-summary__list">
      <div class="prep-summary__row">
        <dt>Folder</dt>
        <dd>
          <span v-if="folderName">{{ folderName }}</span>
          <span v-else class="prep-summary__muted">choose a folder…</span>
        </dd>
      </div>
      <div class="prep-summary__row">
        <dt>Captions</dt>
        <dd>{{ captionFormatLabel }}</dd>
      </div>

      <!-- Tag -->
      <template v-if="stage === 'tag'">
        <div class="prep-summary__row">
          <dt>Models</dt>
          <dd>
            <span v-if="tagForm.models.length">{{ tagForm.models.join(', ') }}</span>
            <span v-else class="prep-summary__muted">none selected</span>
          </dd>
        </div>
        <div v-for="mid in tagForm.models" :key="mid" class="prep-summary__row">
          <dt>{{ mid }}</dt>
          <dd>{{ thresholdSummary(mid) }}</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Max tags</dt>
          <dd>{{ tagForm.max_tags }}</dd>
        </div>
        <div v-if="tagForm.prepend_tags.length || tagForm.exclude_tags.length" class="prep-summary__row">
          <dt>Tag edits</dt>
          <dd>{{ tagForm.prepend_tags.length }} prepend · {{ tagForm.exclude_tags.length }} exclude</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Overwrite</dt>
          <dd>{{ tagForm.overwrite ? 'yes' : 'no' }}</dd>
        </div>
        <div v-if="tagForm.quality_tags" class="prep-summary__row">
          <dt>Quality tags</dt>
          <dd>on</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Tag form</dt>
          <dd>{{ tagForm.underscores ? 'underscores (long_hair)' : 'spaces (long hair)' }}</dd>
        </div>
      </template>

      <!-- Caption -->
      <template v-else-if="stage === 'caption'">
        <div class="prep-summary__row">
          <dt>Model</dt>
          <dd>
            <span v-if="captionForm.model && captionForm.engine === 'vllm'">{{ captionForm.model }} · vllm/{{ captionForm.vllm_quantization }}</span>
            <span v-else-if="captionForm.model">{{ captionForm.model }} · {{ captionForm.quantization }}</span>
            <span v-else class="prep-summary__muted">not selected</span>
          </dd>
        </div>
        <div v-if="vramHint" class="prep-summary__row">
          <dt>VRAM</dt>
          <dd>{{ vramHint }}</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Prompt base</dt>
          <dd>{{ promptBaseLabel }}</dd>
        </div>
        <div v-if="modifierLabels.length" class="prep-summary__row">
          <dt>Modifiers</dt>
          <dd>{{ modifierLabels.join(', ') }}</dd>
        </div>
        <div v-if="captionForm.character_name.trim()" class="prep-summary__row">
          <dt>Character</dt>
          <dd>{{ captionForm.character_name }} · {{ captionForm.outfit }}</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Caption line</dt>
          <dd>{{ captionForm.target_line }}</dd>
        </div>
      </template>

      <!-- Clean -->
      <template v-else-if="stage === 'clean'">
        <div class="prep-summary__row">
          <dt>Confidence</dt>
          <dd>{{ cleanForm.confidence }}</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Mask dilation</dt>
          <dd>{{ cleanForm.mask_dilation_px }} px</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Output</dt>
          <dd>{{ cleanForm.in_place ? 'in-place (overwrite)' : (cleanForm.output_dir || '<dataset>/cleaned') }}</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Undetected</dt>
          <dd>{{ cleanForm.copy_undetected ? 'copied' : 'skipped' }}</dd>
        </div>
      </template>

      <!-- Quality -->
      <template v-else-if="stage === 'quality'">
        <div class="prep-summary__row">
          <dt>Metric</dt>
          <dd>{{ qualityForm.metric }}</dd>
        </div>
        <div v-if="qualityForm.metric === 'blur'" class="prep-summary__row">
          <dt>Blur floor</dt>
          <dd>{{ qualityForm.blur_threshold }}</dd>
        </div>
        <div v-if="qualityForm.metric === 'blur' && qualityForm.min_side > 0" class="prep-summary__row">
          <dt>Min side</dt>
          <dd>{{ qualityForm.min_side }} px</dd>
        </div>
        <div v-if="qualityForm.metric === 'blur' && qualityForm.min_detail > 0" class="prep-summary__row">
          <dt>Min detail</dt>
          <dd>{{ qualityForm.min_detail }}</dd>
        </div>
        <div v-if="qualityForm.metric === 'aesthetic'" class="prep-summary__row">
          <dt>Min label</dt>
          <dd>{{ qualityForm.aesthetic_min_label }}</dd>
        </div>
        <div v-if="qualityForm.metric === 'iqa'" class="prep-summary__row">
          <dt>IQA model</dt>
          <dd>{{ qualityForm.iqa_model }}</dd>
        </div>
        <div v-if="qualityForm.metric === 'iqa'" class="prep-summary__row">
          <dt>Cull lowest %</dt>
          <dd>{{ qualityForm.iqa_threshold }}%</dd>
        </div>
        <div class="prep-summary__row">
          <dt>Action</dt>
          <dd>{{ qualityForm.move ? 'move to low_quality/' : 'report only' }}</dd>
        </div>
        <div v-if="qualityForm.move && qualityForm.output_dir" class="prep-summary__row">
          <dt>Output dir</dt>
          <dd>{{ qualityForm.output_dir }}</dd>
        </div>
      </template>
    </dl>

    <template v-if="stage === 'caption'">
      <el-divider class="prep-summary__divider" content-position="left">
        Prompt preview
        <el-tag v-if="previewNative" size="small" type="warning" class="ml-8">native ToriiGate format</el-tag>
      </el-divider>
      <pre class="prep-summary__preview">{{ previewText || '(waiting for model selection…)' }}</pre>
    </template>

    <el-text size="small" type="info" class="prep-summary__note">
      Jobs run one at a time; extra jobs queue in FIFO order.
    </el-text>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { PropType } from "vue";
import type { PrepPromptOptions, PrepStage } from "../types/api";

interface CommonForm {
  path: string;
  caption_format: "sidecar" | "json";
  caption_ext: string;
}
interface TagForm {
  models: string[];
  exclude_tags: string[];
  prepend_tags: string[];
  max_tags: number;
  overwrite: boolean;
  quality_tags: boolean;
  underscores: boolean;
}
interface ModelThresholds {
  general: number;
  character: number;
  rating: number;
}
interface CaptionForm {
  model: string;
  quantization: "bf16" | "int8" | "nf4";
  prompt_base: string;
  prompt_modifiers: string[];
  character_name: string;
  outfit: string;
  target_line: number;
  engine: "hf" | "vllm";
  vllm_quantization: "gptq" | "fp8" | "awq" | "none";
}
interface CleanForm {
  confidence: number;
  mask_dilation_px: number;
  in_place: boolean;
  output_dir: string;
  copy_undetected: boolean;
}
interface QualityForm {
  metric: "blur" | "aesthetic" | "iqa";
  blur_threshold: number;
  min_side: number;
  min_detail: number;
  aesthetic_min_label: string;
  iqa_model: string;
  iqa_threshold: number;
  move: boolean;
  output_dir: string;
}

const props = defineProps({
  stage: { type: String as PropType<PrepStage>, required: true },
  form: { type: Object as PropType<CommonForm>, required: true },
  tagForm: { type: Object as PropType<TagForm>, required: true },
  tagThresholds: { type: Object as PropType<Record<string, ModelThresholds>>, default: () => ({}) },
  captionForm: { type: Object as PropType<CaptionForm>, required: true },
  cleanForm: { type: Object as PropType<CleanForm>, required: true },
  qualityForm: { type: Object as PropType<QualityForm>, required: true },
  promptOptions: { type: Object as PropType<PrepPromptOptions | null>, default: null },
  previewText: { type: String, default: "" },
  previewNative: { type: Boolean, default: false },
});

function basename(p: string): string {
  if (!p) return "";
  return p.replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() || p;
}

const folderName = computed(() => basename(props.form.path.trim()));

const captionFormatLabel = computed(() =>
  props.form.caption_format === "json"
    ? "captions.json"
    : `sidecar (${props.form.caption_ext || ".txt"})`
);

function thresholdSummary(modelId: string): string {
  const t = props.tagThresholds[modelId];
  if (!t) return "—";
  const c = t.character > 0 ? t.character : "off";
  const r = t.rating > 0 ? t.rating : "off";
  return `g ${t.general} · c ${c} · r ${r}`;
}

const vramHint = computed(() => {
  if (props.captionForm.engine === "vllm") {
    // vLLM weights footprint by quantization (KV cache fills the rest of the budget).
    switch (props.captionForm.vllm_quantization) {
      case "gptq":
      case "awq":
        return "~5 GB (fits 16 GB)";
      case "fp8":
        return "~8.5 GB";
      case "none":
        return "~17 GB (24 GB card)";
      default:
        return "";
    }
  }
  const isTorii = props.captionForm.model === "toriigate-0.5";
  switch (props.captionForm.quantization) {
    case "bf16":
      return isTorii ? "~10 GB" : "~17 GB";
    case "int8":
      return "≈ half of bf16";
    case "nf4":
      return "smallest footprint";
    default:
      return "";
  }
});

const promptBaseLabel = computed(() => {
  const base = props.promptOptions?.bases.find((b) => b.id === props.captionForm.prompt_base);
  return base?.label ?? props.captionForm.prompt_base;
});

const modifierLabels = computed(() => {
  const mods = props.promptOptions?.modifiers ?? [];
  return props.captionForm.prompt_modifiers.map(
    (id) => mods.find((m) => m.id === id)?.label ?? id
  );
});
</script>

<style scoped>
.prep-summary-card {
  font-size: 13px;
}
.prep-summary__list {
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: var(--rf-space-xs);
}
.prep-summary__row {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 8px;
  align-items: baseline;
}
.prep-summary__row dt {
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
.prep-summary__row dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.prep-summary__muted {
  color: var(--el-text-color-placeholder);
}
.prep-summary__divider {
  margin: var(--rf-space-md) 0 var(--rf-space-sm);
}
.prep-summary__preview {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  line-height: 1.5;
  background: var(--el-fill-color-light);
  border-radius: 4px;
  padding: 8px 10px;
  margin: 0;
  max-height: 320px;
  overflow-y: auto;
  color: var(--el-text-color-secondary);
}
.prep-summary__note {
  display: block;
  margin-top: var(--rf-space-md);
}
.ml-8 {
  margin-left: 8px;
}
</style>

<!--
  The `folder` node's config: the three fields that make up a dataset handle.

  This is the one node whose config *is* the handle — every other node inherits those three from
  its incoming edge — so it reuses `PrepCommonFields.vue` unhidden, with the same folder-existence
  validation `PathFieldControl` gives every path field in the app, plus a live media count so the
  user can tell "the folder exists" from "the folder has images in it".
-->
<template>
  <div class="folder-node-form">
    <PrepCommonFields v-model="form" stage="tag" :disabled="disabled" />

    <div class="folder-node-form__stats">
      <PathValidationFeedback :loading="statsLoading" :error="statsError" />
      <el-text v-if="stats && stats.ok !== false" size="small" type="info">
        {{ mediaSummary }}
      </el-text>
    </div>

    <el-alert
      type="info"
      :closable="false"
      show-icon
      class="folder-node-form__note"
      title="Every later step reads this folder through its incoming edge, so changing the path here changes the whole chain's input in one edit."
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import PrepCommonFields from "../../prep/PrepCommonFields.vue";
import PathValidationFeedback from "../../PathValidationFeedback.vue";
import { useDatasetFolderStats } from "../../../composables/useDatasetFolderStats";
import { defaultCommonForm, type PrepCommonForm } from "../../../lib/prepStageConfig";

const config = defineModel<Record<string, unknown>>({ required: true });

defineProps({
  /** Read-only: the runner owns the workflow while it runs. */
  disabled: { type: Boolean, default: false },
});

function fromConfig(source: Record<string, unknown>): PrepCommonForm {
  const base = defaultCommonForm();
  return {
    path: typeof source.path === "string" ? source.path : base.path,
    caption_format: source.caption_format === "json" ? "json" : "sidecar",
    caption_ext: typeof source.caption_ext === "string" ? source.caption_ext : base.caption_ext,
  };
}

function sameAsForm(source: Record<string, unknown>): boolean {
  const next = fromConfig(source);
  return (
    next.path === form.value.path &&
    next.caption_format === form.value.caption_format &&
    next.caption_ext === form.value.caption_ext
  );
}

/**
 * `ref`, not `reactive`: `v-model` on a `const reactive(...)` binding cannot compile a setter, so
 * the compiler warns on every build. The object is still mutated in place — a `ref`'s value is
 * deeply reactive all the same.
 */
const form = ref<PrepCommonForm>(fromConfig(config.value));

// The two watchers cannot loop: each one bails as soon as the other side already agrees.
watch(
  () => config.value,
  (next) => {
    if (!sameAsForm(next)) Object.assign(form.value, fromConfig(next));
  },
);

watch(
  form,
  () => {
    if (sameAsForm(config.value)) return;
    config.value = {
      ...config.value,
      path: form.value.path,
      caption_format: form.value.caption_format,
      caption_ext: form.value.caption_ext,
    };
  },
  { deep: true },
);

const { loading: statsLoading, error: statsError, stats, load, clear } = useDatasetFolderStats();

let statsTimer: ReturnType<typeof setTimeout> | null = null;
watch(
  () => form.value.path,
  (path) => {
    if (statsTimer) clearTimeout(statsTimer);
    const trimmed = (path || "").trim();
    if (!trimmed) {
      clear();
      return;
    }
    statsTimer = setTimeout(() => void load(trimmed), 500);
  },
  { immediate: true },
);

const mediaSummary = computed(() => {
  const data = stats.value;
  if (!data) return "";
  const images = data.image_count_display ?? String(data.image_count ?? 0);
  const parts = [`${images} images`];
  if (data.video_count) parts.push(`${data.video_count} videos`);
  if (data.has_captions_json) parts.push("captions.json");
  else if (data.caption_txt_files) parts.push(`${data.caption_txt_files} caption files`);
  return parts.join(" · ");
});
</script>

<style scoped>
.folder-node-form__stats {
  min-height: 20px;
  margin-top: -8px;
  margin-bottom: 12px;
}
.folder-node-form__note {
  margin-top: 4px;
}
</style>

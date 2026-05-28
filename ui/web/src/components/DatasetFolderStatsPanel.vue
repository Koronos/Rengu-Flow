<template>
  <div class="folder-stats" @click.stop>
    <p class="folder-stats-title">Folder stats</p>
    <p v-if="path" class="folder-stats-path" :title="path">{{ path }}</p>
    <div v-if="loading" class="folder-stats-loading">
      <el-skeleton :rows="2" animated />
    </div>
    <el-alert
      v-else-if="error"
      type="warning"
      :title="error"
      show-icon
      :closable="false"
    />
    <dl v-else-if="stats?.ok" class="folder-stats-dl">
      <div class="folder-stats-row">
        <dt>Images</dt>
        <dd>{{ formatMediaCount(stats.image_count, capped) }}</dd>
      </div>
      <div v-if="videoCount > 0" class="folder-stats-row">
        <dt>Videos</dt>
        <dd>{{ formatMediaCount(stats.video_count, capped) }}</dd>
      </div>
      <div class="folder-stats-row">
        <dt>Paired .txt captions</dt>
        <dd>{{ formatMediaCount(stats.caption_txt_files, capped) }}</dd>
      </div>
      <div v-if="stats.has_captions_json" class="folder-stats-row">
        <dt>captions.json</dt>
        <dd>present</dd>
      </div>
    </dl>
    <p class="folder-stats-note">
      Non-recursive scan of the folder path (same as training). Caption .txt files must share the image base name.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { formatMediaCount } from "../lib/formatMediaCount";
import type { DatasetScanPathResult } from "../types/api";

const props = defineProps({
  path: { type: String, default: "" },
  loading: { type: Boolean, default: false },
  error: { type: String, default: "" },
  stats: { type: Object as () => DatasetScanPathResult | null, default: null },
});

const capped = computed(() => Boolean(props.stats?.count_capped));
const videoCount = computed(() => Number(props.stats?.video_count || 0));
</script>

<style scoped>
.folder-stats-title {
  margin: 0 0 6px;
  font-size: 13px;
  font-weight: 600;
}
.folder-stats-path {
  margin: 0 0 10px;
  font-size: 11px;
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  color: var(--el-text-color-secondary);
  word-break: break-all;
  line-height: 1.35;
}
.folder-stats-loading {
  margin-bottom: 8px;
}
.folder-stats-dl {
  margin: 0 0 8px;
}
.folder-stats-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  line-height: 1.6;
}
.folder-stats-row dt {
  margin: 0;
  color: var(--el-text-color-secondary);
}
.folder-stats-row dd {
  margin: 0;
  font-variant-numeric: tabular-nums;
}
.folder-stats-note {
  margin: 0;
  font-size: 11px;
  color: var(--el-text-color-placeholder);
  line-height: 1.4;
}
</style>

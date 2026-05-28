<template>
  <el-popover
    placement="left"
    :width="280"
    trigger="click"
    :disabled="disabled"
    @before-enter="onShow"
    @after-leave="onHide"
  >
    <template #reference>
      <span class="folder-stats-ref" @click.stop>
        <el-tooltip content="Folder stats" :show-after="300">
          <el-button size="small" circle :icon="DataAnalysis" :disabled="disabled" />
        </el-tooltip>
      </span>
    </template>
    <DatasetFolderStatsPanel
      :path="path"
      :loading="loading"
      :error="error"
      :stats="stats"
    />
  </el-popover>
</template>

<script setup lang="ts">
import { DataAnalysis } from "@element-plus/icons-vue";
import DatasetFolderStatsPanel from "./DatasetFolderStatsPanel.vue";
import { useDatasetFolderStats } from "../composables/useDatasetFolderStats";

const props = defineProps({
  path: { type: String, default: "" },
  disabled: { type: Boolean, default: false },
});

const { loading, error, stats, load, clear } = useDatasetFolderStats();

async function onShow(): Promise<void> {
  await load(props.path);
}

function onHide(): void {
  clear();
}
</script>

<style scoped>
.folder-stats-ref {
  display: inline-flex;
  align-items: center;
}
</style>

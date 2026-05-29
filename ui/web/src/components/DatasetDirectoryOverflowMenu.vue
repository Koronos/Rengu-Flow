<template>
  <div class="dp-dir-overflow" @click.stop>
    <span ref="statsAnchorRef" class="dp-dir-stats-anchor" aria-hidden="true" />
    <el-tooltip content="More actions" :show-after="300">
      <el-dropdown trigger="click" @command="onCommand">
        <el-button
          class="dp-dir-overflow-trigger"
          size="small"
          circle
          :icon="MoreFilled"
          @click.stop
        />
        <template #dropdown>
          <el-dropdown-item command="gallery" :disabled="galleryDisabled">
            <span class="rf-dropdown-item-label">
              <el-icon><Picture /></el-icon>
              <span>Image gallery</span>
            </span>
          </el-dropdown-item>
          <el-dropdown-item command="stats" :disabled="statsDisabled">
            <span class="rf-dropdown-item-label">
              <el-icon><DataAnalysis /></el-icon>
              <span>Folder stats</span>
            </span>
          </el-dropdown-item>
          <el-dropdown-item command="delete" divided>
            <span class="rf-dropdown-item-label rf-dropdown-item-label--danger">
              <el-icon><Delete /></el-icon>
              <span>{{ deleteLabel }}</span>
            </span>
          </el-dropdown-item>
        </template>
      </el-dropdown>
    </el-tooltip>
    <el-popover
      v-model:visible="statsVisible"
      placement="left"
      :width="280"
      trigger="click"
      v-bind="virtualTrigger"
      :disabled="statsDisabled"
      @before-enter="onStatsShow"
      @after-leave="onStatsHide"
    >
      <DatasetFolderStatsPanel
        :path="statsPath"
        :loading="loading"
        :error="error"
        :stats="stats"
      />
    </el-popover>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, shallowRef } from "vue";
import { DataAnalysis, Delete, MoreFilled, Picture } from "@element-plus/icons-vue";
import DatasetFolderStatsPanel from "./DatasetFolderStatsPanel.vue";
import { useDatasetFolderStats } from "../composables/useDatasetFolderStats";

const props = defineProps({
  statsPath: { type: String, default: "" },
  galleryDisabled: { type: Boolean, default: false },
  statsDisabled: { type: Boolean, default: false },
  deleteLabel: { type: String, default: "Remove directory" },
});

const emit = defineEmits<{
  gallery: [];
  delete: [];
}>();

const statsAnchorRef = shallowRef<HTMLElement | null>(null);
const statsVisible = ref(false);
const { loading, error, stats, load, clear } = useDatasetFolderStats();

// el-popover's prop types omit `virtual-triggering`/`virtual-ref`; forward them as fallthrough attrs.
const virtualTrigger = computed<Record<string, unknown>>(() => ({
  "virtual-triggering": true,
  "virtual-ref": statsAnchorRef.value,
}));

function onCommand(command: string | number) {
  if (command === "gallery") emit("gallery");
  else if (command === "delete") emit("delete");
  else if (command === "stats" && !props.statsDisabled) {
    statsVisible.value = true;
  }
}

async function onStatsShow(): Promise<void> {
  await load(props.statsPath);
}

function onStatsHide(): void {
  clear();
}
</script>

<style scoped>
.dp-dir-overflow {
  position: relative;
  display: inline-flex;
  align-items: center;
}
.dp-dir-stats-anchor {
  position: absolute;
  right: 0;
  top: 50%;
  width: 1px;
  height: 1px;
  margin-top: -1px;
  pointer-events: none;
}
</style>

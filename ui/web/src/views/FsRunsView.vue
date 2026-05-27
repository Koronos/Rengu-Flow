<template>
  <div>
    <h2 v-if="!embedded" class="page-title">Runs</h2>
    <p v-if="!embedded" class="page-subtitle">All output folders under output_dir</p>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <el-card shadow="never" class="mb-12">
      <el-form inline :class="{ 'mobile-form': isMobile }">
        <el-form-item label="output_dir">
          <el-input v-model="outputDir" clearable style="min-width: 160px" />
        </el-form-item>
        <el-form-item>
          <el-button :icon="Refresh" @click="load">Refresh</el-button>
          <el-button :loading="tbLoading" @click="openTensorboardForOutput">Open TensorBoard</el-button>
          <el-button
            v-if="tbStatus?.running"
            :loading="tbLoading"
            type="danger"
            plain
            @click="stopTensorboardForOutput"
          >
            Stop TensorBoard
          </el-button>
          <el-link
            v-if="tbStatus?.running && tbStatus.url"
            :href="tbStatus.url"
            target="_blank"
            type="primary"
          >
            {{ tbStatus.url }}
          </el-link>
        </el-form-item>
      </el-form>
    </el-card>

    <el-table
      :data="runs"
      stripe
      style="width: 100%"
      size="small"
      @row-click="onRowClick"
    >
      <el-table-column prop="name" label="Name" min-width="160">
        <template #default="{ row }">
          <el-link type="primary" :underline="false" @click.stop="goRun(row.name)">
            {{ row.name }}
          </el-link>
        </template>
      </el-table-column>
      <el-table-column label="TensorBoard" width="110">
        <template #default="{ row }">
          <el-tag v-if="row.has_tensorboard" type="success" size="small">yes</el-tag>
          <span v-else>—</span>
        </template>
      </el-table-column>
      <el-table-column label="Status" min-width="120">
        <template #default="{ row }">
          <span v-if="row.status">step {{ row.status.step }}</span>
          <span v-else>—</span>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { Refresh } from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";
import { useTensorboard } from "../composables/useTensorboard";
import type { JsonRecord } from "../types/runtime";

defineProps({
  embedded: { type: Boolean, default: false },
});

const router = useRouter();
const { isMobile } = useBreakpoint();

const runs = ref<JsonRecord[]>([]);
const outputDir = ref("output");
const error = ref("");
const { tbLoading, tbStatus, refreshTbStatus, openTensorboard, stopTensorboard } = useTensorboard(
  () => outputDir.value || "output"
);

async function load() {
  const data = (await api.listFsRuns(outputDir.value)) as { runs?: JsonRecord[] };
  runs.value = data.runs || [];
}

function openTensorboardForOutput() {
  openTensorboard({ onError: (msg) => { error.value = msg; } }).catch(() => {});
}

function stopTensorboardForOutput() {
  stopTensorboard({ onError: (msg) => { error.value = msg; } }).catch(() => {});
}

function goRun(name) {
  router.push(`/runs/${encodeURIComponent(name)}`);
}

function onRowClick(row) {
  goRun(row.name);
}

onMounted(() => {
  refreshTbStatus();
  load().catch((e) => { error.value = String(e); });
});

watch(outputDir, () => {
  load().catch((e) => { error.value = String(e); });
});
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
.mobile-form :deep(.el-form-item) {
  display: block;
  margin-right: 0;
}
</style>

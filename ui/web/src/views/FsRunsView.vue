<template>
  <div>
    <h2 class="page-title">Output runs</h2>

    <el-alert v-if="error" type="error" :title="error" show-icon class="mb-12" />

    <el-card shadow="never" class="mb-12">
      <el-form inline :class="{ 'mobile-form': isMobile }">
        <el-form-item label="output_dir">
          <el-input v-model="outputDir" clearable style="min-width: 160px" />
        </el-form-item>
        <el-form-item>
          <el-button :icon="Refresh" @click="load">Refresh</el-button>
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

<script setup>
import { onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import { Refresh } from "@element-plus/icons-vue";
import { api } from "../api";
import { useBreakpoint } from "../composables/useBreakpoint";

const router = useRouter();
const { isMobile } = useBreakpoint();

const runs = ref([]);
const outputDir = ref("output");
const error = ref("");

async function load() {
  const data = await api.listFsRuns(outputDir.value);
  runs.value = data.runs || [];
}

function goRun(name) {
  router.push(`/runs/${encodeURIComponent(name)}`);
}

function onRowClick(row) {
  goRun(row.name);
}

onMounted(() => {
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

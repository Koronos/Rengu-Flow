<template>
  <div class="toolbox-view">
    <div class="toolbar">
      <h2>Toolbox</h2>
      <el-button type="primary" @click="$router.push('/toolbox/new')">New tool</el-button>
    </div>
    <el-alert
      v-if="!executionEnabled"
      type="info"
      :closable="false"
      title="Execution disabled in rengu.local.toml → [toolbox].enabled. You can still create and edit tools."
    />
    <el-empty v-if="tools.length === 0" description="No tools yet" />
    <div v-else class="tool-grid">
      <el-card v-for="t in tools" :key="t.id" class="tool-card">
        <div class="tool-card__head">
          <strong>{{ t.name }}</strong>
          <el-tag size="small" :type="statusTagType(t.last_run_status)">{{ t.last_run_status }}</el-tag>
        </div>
        <p class="tool-card__desc">{{ t.description }}</p>
        <p class="tool-card__dates">
          Created {{ t.created_at }} · Modified {{ t.updated_at }}
        </p>
        <div class="tool-card__actions">
          <el-button size="small" @click="$router.push(`/toolbox/${t.id}/edit`)">Edit</el-button>
          <el-button size="small" type="danger" @click="remove(t.id)">Delete</el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessageBox } from "element-plus";
import { api, type ToolboxToolSummary } from "../api";

const tools = ref<ToolboxToolSummary[]>([]);
const executionEnabled = ref(true);

function statusTagType(s: string): "success" | "danger" | "warning" | "info" {
  return s === "done" ? "success" : s === "failed" ? "danger" : s === "running" ? "warning" : "info";
}

async function load() {
  tools.value = await api.listToolboxTools();
  executionEnabled.value = (await api.toolboxEnabled()).enabled;
}

async function remove(id: string) {
  await ElMessageBox.confirm("Delete this tool?", "Confirm", { type: "warning" });
  await api.deleteToolboxTool(id);
  await load();
}

onMounted(load);
</script>

<template>
  <div class="toolbox-view">
    <div class="toolbar">
      <el-input
        v-model="query"
        class="toolbar__search"
        placeholder="Search tools"
        :prefix-icon="Search"
        clearable
      />
      <el-button type="primary" :icon="Plus" @click="$router.push('/toolbox/new')">
        New tool
      </el-button>
    </div>

    <el-alert
      v-if="!executionEnabled"
      type="info"
      class="toolbox-view__banner"
      :closable="false"
      show-icon
      title="Execution disabled"
      description="Set [toolbox].enabled = true in rengu.local.toml to run tools. You can still create and edit them."
    />

    <el-empty
      v-if="tools.length === 0"
      description="No tools yet — create one to run your own Python scripts"
    />
    <el-empty
      v-else-if="filtered.length === 0"
      :description="`No tools match “${query}”`"
    />
    <div v-else class="tool-grid">
      <article
        v-for="t in filtered"
        :key="t.id"
        class="tool-card"
        tabindex="0"
        @click="edit(t.id)"
        @keydown.enter="edit(t.id)"
      >
        <header class="tool-card__head">
          <span class="tool-card__name">{{ t.name }}</span>
          <el-tag size="small" effect="dark" :type="statusTagType(t.last_run_status)">
            {{ t.last_run_status }}
          </el-tag>
        </header>
        <p class="tool-card__desc">{{ t.description || "No description" }}</p>
        <footer class="tool-card__foot">
          <span class="tool-card__date">Updated {{ fmtDate(t.updated_at) }}</span>
          <el-tooltip content="Delete" placement="top">
            <el-button
              text
              size="small"
              :icon="Delete"
              class="tool-card__del"
              @click.stop="remove(t.id)"
            />
          </el-tooltip>
        </footer>
      </article>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { ElMessageBox } from "element-plus";
import { Delete, Plus, Search } from "@element-plus/icons-vue";
import { api, type ToolboxToolSummary } from "../api";

const router = useRouter();
const tools = ref<ToolboxToolSummary[]>([]);
const executionEnabled = ref(true);
const query = ref("");

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase();
  if (!q) return tools.value;
  return tools.value.filter(
    (t) =>
      t.name.toLowerCase().includes(q) || (t.description || "").toLowerCase().includes(q),
  );
});

function statusTagType(s: string): "success" | "danger" | "warning" | "info" {
  return s === "done" ? "success" : s === "failed" ? "danger" : s === "running" ? "warning" : "info";
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function edit(id: string) {
  router.push(`/toolbox/${id}/edit`);
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

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: var(--rf-space-sm);
  margin-bottom: var(--rf-space-md);
}
.toolbar__search {
  max-width: 360px;
}
.toolbox-view__banner {
  margin-bottom: var(--rf-space-md);
}

.tool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: var(--rf-space-sm);
}
.tool-card {
  display: flex;
  flex-direction: column;
  gap: var(--rf-space-xs);
  padding: var(--rf-space-sm);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  background: var(--el-bg-color);
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.tool-card:hover,
.tool-card:focus-visible {
  border-color: var(--el-color-primary);
  transform: translateY(-1px);
  outline: none;
}
.tool-card__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--rf-space-xs);
}
.tool-card__name {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tool-card__desc {
  margin: 0;
  color: var(--el-text-color-secondary);
  font-size: 13px;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 2.8em;
}
.tool-card__foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: auto;
  padding-top: var(--rf-space-xs);
  border-top: 1px solid var(--el-border-color-lighter);
}
.tool-card__date {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.tool-card__del {
  color: var(--el-text-color-secondary);
}
.tool-card__del:hover {
  color: var(--el-color-danger);
}
</style>

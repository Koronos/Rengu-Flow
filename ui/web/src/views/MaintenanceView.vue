<template>
  <div class="maintenance-view page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Local development tools (destructive actions)</p>
      </div>
    </div>

    <el-alert
      v-if="!enabled"
      type="warning"
      show-icon
      class="mb-12"
      title="Maintenance disabled"
      description="Set RENGAFLOW_MAINTENANCE=1 on the control server and restart (e.g. in start-ui.sh), then reload this page."
    />

    <el-alert
      v-else
      type="error"
      show-icon
      class="mb-12"
      title="Destructive operations"
      description="Recreating the database deletes all saved configs, datasets, and job history in jobs.db. Use only on a dev machine."
    />

    <el-skeleton v-if="loading" :rows="6" animated />

    <template v-else-if="status">
      <el-row :gutter="16" class="mb-12">
        <el-col :xs="24" :md="12">
          <el-card shadow="never">
            <template #header>Database</template>
            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="Path">{{ status.database.path }}</el-descriptions-item>
              <el-descriptions-item label="Size">
                {{ formatBytes(status.database.size_bytes) }}
              </el-descriptions-item>
              <el-descriptions-item label="Modified">
                {{ formatMtime(status.database.modified_at) }}
              </el-descriptions-item>
              <el-descriptions-item label="Tables">
                {{ status.database.tables.join(", ") || "(none)" }}
              </el-descriptions-item>
            </el-descriptions>
            <el-tooltip
              content="Deletes jobs.db and recreates empty tables. All saved configs, datasets, and job history are lost."
              placement="top"
              :show-after="300"
            >
              <el-button
                type="danger"
                class="mt-12"
                :disabled="!enabled || busy"
                @click="onResetDatabase"
              >
                Recreate database
              </el-button>
            </el-tooltip>
          </el-card>
        </el-col>

        <el-col :xs="24" :md="12">
          <el-card shadow="never">
            <template #header>Git submodules</template>
            <p v-if="!status.git.gitmodules_exists" class="muted-text">
              This repository has no <code>.gitmodules</code> (Cosmos code is vendored in-tree).
              Submodule init is only relevant for upstream diffusion-pipe clones.
            </p>
            <ul v-else class="submodule-list">
              <li v-for="row in status.git.submodule_status" :key="row.path">
                <code>{{ row.line }}</code>
              </li>
            </ul>
            <el-tooltip
              content="Runs git submodule init/update for upstream diffusion-pipe clones. No effect when Cosmos code is vendored in-tree."
              placement="top"
              :show-after="300"
            >
              <el-button
                type="primary"
                plain
                class="mt-12"
                :disabled="!enabled || busy"
                @click="onSubmodulesUpdate"
              >
                Initialize / update submodules
              </el-button>
            </el-tooltip>
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" class="mb-12">
        <template #header>Dependencies</template>
        <p class="muted-text mb-8">
          Install into the same Python environment as the control server
          (<code>{{ status.python_executable }}</code>). Cosmos training needs the
          <strong>cosmos_predict2</strong> extra separately from the base stack.
        </p>
        <el-table :data="status.dependency_profiles" size="small" stripe>
          <el-table-column prop="label" label="Profile" width="160" />
          <el-table-column prop="description" label="Description" />
          <el-table-column label="Command" min-width="240">
            <template #default="{ row }">
              <code class="dep-cmd">{{ row.command }}</code>
            </template>
          </el-table-column>
          <el-table-column label="" width="200" align="right">
            <template #default="{ row }">
              <el-button size="small" @click="copyCommand(row.command)">Copy</el-button>
              <el-button
                v-if="status.pip_install_from_server"
                size="small"
                type="warning"
                :disabled="!enabled || busy"
                @click="onDepsInstall(row.id, true)"
              >
                Run
              </el-button>
            </template>
          </el-table-column>
        </el-table>
        <el-alert
          v-if="!status.pip_install_from_server"
          type="info"
          :closable="false"
          class="mt-12"
          title="Server pip disabled"
          description="Copy commands into your venv. To allow Run from the UI, set RENGAFLOW_MAINTENANCE_ALLOW_PIP=1 on the server."
        />
      </el-card>

      <el-card shadow="never" class="log-card">
        <template #header>
          <span>Command output</span>
          <el-button v-if="logText" text type="primary" size="small" @click="clearLog">Clear</el-button>
        </template>
        <pre class="log-panel">{{ logText || "(no output yet)" }}</pre>
      </el-card>
    </template>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import { api } from "../api";
import { formatError } from "../lib/formatError";
import type { MaintenanceStatus } from "../types/api";

const enabled = ref(false);
const loading = ref(true);
const busy = ref(false);
const status = ref<MaintenanceStatus | null>(null);
const logText = ref("");

function appendLog(label: string, payload: { stdout?: string; stderr?: string; message?: string }) {
  const parts = [label];
  if (payload.message) parts.push(payload.message);
  if (payload.stdout) parts.push(payload.stdout);
  if (payload.stderr) parts.push(payload.stderr);
  logText.value = `${logText.value}\n${parts.join("\n")}\n`.trimStart();
}

function clearLog() {
  logText.value = "";
}

function formatBytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatMtime(ts: number | null): string {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString();
}

async function load() {
  loading.value = true;
  try {
    const en = await api.maintenanceEnabled();
    enabled.value = en.enabled;
    if (en.enabled) {
      status.value = await api.maintenanceStatus();
    } else {
      status.value = null;
    }
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    loading.value = false;
  }
}

async function onResetDatabase() {
  try {
    await ElMessageBox.confirm(
      'Type RESET in the box below to wipe jobs.db and recreate empty tables. All configs, datasets, and job history will be lost.',
      "Recreate database",
      {
        confirmButtonText: "Recreate",
        cancelButtonText: "Cancel",
        type: "warning",
        inputPattern: /^RESET$/,
        inputErrorMessage: 'Type RESET exactly',
        showInput: true,
        inputPlaceholder: "RESET",
      }
    );
  } catch {
    return;
  }
  busy.value = true;
  try {
    const res = await api.maintenanceDatabaseReset();
    appendLog("Database reset", res);
    ElMessage.success("Database recreated");
    status.value = await api.maintenanceStatus();
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    busy.value = false;
  }
}

async function onSubmodulesUpdate() {
  try {
    await ElMessageBox.confirm(
      "Run git submodule update --init --recursive in the repository root?",
      "Update submodules",
      { type: "info", confirmButtonText: "Run", cancelButtonText: "Cancel" }
    );
  } catch {
    return;
  }
  busy.value = true;
  try {
    const res = await api.maintenanceSubmodulesUpdate();
    appendLog("Submodules", res);
    if (res.ok) ElMessage.success(res.message || "Done");
    else ElMessage.warning(res.message || "Submodule command failed");
    status.value = await api.maintenanceStatus();
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    busy.value = false;
  }
}

async function onDepsInstall(profile: string, execute: boolean) {
  if (execute) {
    try {
      await ElMessageBox.confirm(
        `Run pip for profile "${profile}" in the server process? This may take several minutes.`,
        "Install dependencies",
        { type: "warning", confirmButtonText: "Install", cancelButtonText: "Cancel" }
      );
    } catch {
      return;
    }
  }
  busy.value = true;
  try {
    const res = await api.maintenanceDepsInstall(profile, execute);
    appendLog(`Deps (${profile})`, res);
    if (res.executed) {
      if (res.ok) ElMessage.success(res.message || "Install finished");
      else ElMessage.error(res.message || "Install failed");
    } else {
      await copyCommand(res.command);
      ElMessage.info("Command copied to clipboard");
    }
  } catch (e) {
    ElMessage.error(formatError(e));
  } finally {
    busy.value = false;
  }
}

async function copyCommand(cmd: string) {
  try {
    await navigator.clipboard.writeText(cmd);
    ElMessage.success("Copied");
  } catch {
    ElMessage.info(cmd);
  }
}

onMounted(() => {
  load();
});
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
.mt-12 {
  margin-top: 12px;
}
.mb-8 {
  margin-bottom: 8px;
}
.muted-text {
  color: var(--el-text-color-secondary);
  font-size: 13px;
  margin: 0;
}
.submodule-list {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 12px;
}
.dep-cmd {
  font-size: 11px;
  word-break: break-all;
}
.log-panel {
  margin: 0;
  max-height: 280px;
  overflow: auto;
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.log-card :deep(.el-card__header) {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>

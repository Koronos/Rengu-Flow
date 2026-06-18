<template>
  <div class="run-panel">
    <div class="run-panel__head">
      <h3>Run</h3>
      <el-tag v-if="status" :type="statusType" size="small" effect="dark">{{ status }}</el-tag>
    </div>

    <el-alert
      v-if="!enabled"
      type="info"
      class="run-panel__banner"
      :closable="false"
      show-icon
      title="Execution disabled"
      description="Set [toolbox].enabled = true in rengu.local.toml to run tools. You can still edit and save."
    />

    <el-form label-position="top" class="run-form">
      <el-empty v-if="!tool?.inputs?.length" description="No inputs" :image-size="48" />
      <el-form-item
        v-for="inp in tool?.inputs || []"
        :key="inp.param"
        :label="inp.label || inp.param"
      >
        <el-switch v-if="inp.control === 'switch'" v-model="values[inp.param]" />
        <el-input-number
          v-else-if="inp.control === 'number'"
          v-model="values[inp.param]"
          controls-position="right"
        />
        <el-select
          v-else-if="inp.control === 'select'"
          v-model="values[inp.param]"
          placeholder="Select"
        >
          <el-option v-for="o in inp.options || []" :key="o" :label="o" :value="o" />
        </el-select>
        <el-input
          v-else-if="inp.control === 'textarea'"
          v-model="values[inp.param]"
          type="textarea"
          :rows="2"
        />
        <el-input v-else v-model="values[inp.param]" />
        <span v-if="inp.hint" class="hint">{{ inp.hint }}</span>
      </el-form-item>
    </el-form>

    <div class="run-actions">
      <el-button
        type="primary"
        :icon="CaretRight"
        :disabled="!enabled || running"
        :loading="running"
        @click="run"
      >
        {{ running ? "Running…" : "Run" }}
      </el-button>
      <el-button v-if="running" :icon="Close" @click="cancel">Cancel</el-button>
    </div>

    <div class="output">
      <div class="output__bar">
        <span class="output__title">Output</span>
        <span v-if="exitInfo" class="output__meta">{{ exitInfo }}</span>
      </div>
      <pre v-if="cleanLog" class="output__log">{{ cleanLog }}</pre>
      <div v-else class="output__empty">Run the tool to see output here.</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import { CaretRight, Close } from "@element-plus/icons-vue";
import { api, type ToolboxTool } from "../api";
import { wsBaseUrl } from "../lib/wsLog";

const props = defineProps<{ toolId: string }>();

const tool = ref<ToolboxTool | null>(null);
const enabled = ref(true);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const values = reactive<Record<string, any>>({});
const log = ref("");
const status = ref("");
const exitCode = ref<number | null | undefined>(undefined);
const running = computed(() => status.value === "running");
const statusType = computed(() =>
  status.value === "done" ? "success" : status.value === "failed" ? "danger" : "info",
);

// Strip ANSI SGR color codes so the log reads as plain text.
// eslint-disable-next-line no-control-regex
const ANSI_SGR = /\u001b\[[0-9;]*m/g;
const cleanLog = computed(() => log.value.replace(ANSI_SGR, ""));
const exitInfo = computed(() => {
  if (status.value !== "done" && status.value !== "failed") return "";
  return exitCode.value === null || exitCode.value === undefined
    ? status.value
    : `exit ${exitCode.value}`;
});

let ws: WebSocket | null = null;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let pollOffset = 0;

function isTerminal(s: string) {
  return s !== "" && s !== "running" && s !== "idle";
}

function stopPoll() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function loadSnapshot() {
  const r = await api.toolboxLog(props.toolId, 0);
  log.value = r.chunk;
  status.value = r.status;
}

async function pollLog() {
  try {
    const r = await api.toolboxLog(props.toolId, pollOffset);
    if (r.chunk) log.value += r.chunk;
    pollOffset = r.offset;
    status.value = r.status;
    if (isTerminal(r.status)) {
      stopPoll();
      await refreshStatus();
      return;
    }
  } catch {
    // network hiccup — keep polling
  }
  pollTimer = setTimeout(pollLog, 1500);
}

function startPollFallback(fromOffset: number) {
  stopPoll();
  pollOffset = fromOffset;
  pollTimer = setTimeout(pollLog, 1500);
}

function openWs() {
  ws?.close();
  ws = new WebSocket(`${wsBaseUrl()}/api/v1/toolbox/tools/${encodeURIComponent(props.toolId)}/log/ws`);
  let wsOffset = 0;
  ws.onmessage = (ev) => {
    const chunk = ev.data as string;
    log.value += chunk;
    wsOffset += new TextEncoder().encode(chunk).length;
  };
  ws.onerror = () => {
    if (status.value === "running") startPollFallback(wsOffset);
  };
  ws.onclose = () => {
    if (status.value === "running") startPollFallback(wsOffset);
    else refreshStatus();
  };
}

async function refreshStatus() {
  const r = await api.toolboxRunStatus(props.toolId);
  status.value = r.status;
  exitCode.value = r.exit_code;
}

async function run() {
  log.value = "";
  exitCode.value = undefined;
  await api.runToolboxTool(props.toolId, { ...values });
  status.value = "running";
  openWs();
}

async function cancel() {
  await api.cancelToolboxRun(props.toolId);
  await refreshStatus();
}

onMounted(async () => {
  tool.value = await api.getToolboxTool(props.toolId);
  enabled.value = (await api.toolboxEnabled()).enabled;
  for (const inp of tool.value.inputs) {
    if (inp.default !== undefined && inp.default !== null) values[inp.param] = inp.default;
  }
  if (tool.value.last_run?.inputs) Object.assign(values, tool.value.last_run.inputs);
  exitCode.value = tool.value.last_run?.exit_code;
  // Decide how to load the log from the last run's status.
  const lastStatus = tool.value.last_run?.status;
  if (lastStatus === "running") {
    // Stream the live log from the start via WS (no REST snapshot — avoids duplication).
    log.value = "";
    openWs();
  } else {
    // Idle/done/failed: load the final log once via REST.
    await loadSnapshot();
  }
});

onUnmounted(() => {
  ws?.close();
  stopPoll();
});
</script>

<style scoped>
.run-panel {
  border: 1px solid var(--el-border-color);
  border-radius: var(--el-border-radius-base);
  padding: var(--rf-space-md);
  background: var(--el-bg-color);
}
.run-panel__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--rf-space-sm);
}
.run-panel__head h3 {
  margin: 0;
}
.run-panel__banner {
  margin: var(--rf-space-sm) 0;
}
.run-form {
  margin-top: var(--rf-space-sm);
}
.run-actions {
  display: flex;
  gap: var(--rf-space-xs);
  margin: var(--rf-space-sm) 0;
}
.output {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: var(--el-border-radius-base);
  overflow: hidden;
}
.output__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px;
  background: var(--el-fill-color-light);
  border-bottom: 1px solid var(--el-border-color-lighter);
}
.output__title {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--el-text-color-secondary);
}
.output__meta {
  font-size: 12px;
  font-family: var(--rf-font-mono);
  color: var(--el-text-color-secondary);
}
.output__log {
  margin: 0;
  padding: var(--rf-space-sm);
  font-family: var(--rf-font-mono);
  font-size: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 320px;
  overflow: auto;
  color: var(--el-text-color-primary);
}
.output__empty {
  padding: var(--rf-space-md);
  text-align: center;
  font-size: 13px;
  color: var(--el-text-color-secondary);
}
.hint {
  display: block;
  margin-top: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>

<template>
  <div class="run-panel">
    <h3>Run</h3>
    <el-alert
      v-if="!enabled"
      type="info"
      :closable="false"
      title="Execution disabled in rengu.local.toml → [toolbox].enabled"
    />
    <el-form label-position="top">
      <el-form-item v-for="inp in tool?.inputs || []" :key="inp.param" :label="inp.label || inp.param">
        <el-switch v-if="inp.control === 'switch'" v-model="values[inp.param]" />
        <el-input-number v-else-if="inp.control === 'number'" v-model="values[inp.param]" />
        <el-select v-else-if="inp.control === 'select'" v-model="values[inp.param]">
          <el-option v-for="o in inp.options || []" :key="o" :label="o" :value="o" />
        </el-select>
        <el-input v-else-if="inp.control === 'textarea'" v-model="values[inp.param]" type="textarea" />
        <el-input v-else v-model="values[inp.param]" />
        <span v-if="inp.hint" class="hint">{{ inp.hint }}</span>
      </el-form-item>
    </el-form>
    <el-button type="primary" :disabled="!enabled || running" @click="run">Run</el-button>
    <el-button v-if="running" @click="cancel">Cancel</el-button>
    <el-tag v-if="status" :type="statusType" size="small">{{ status }}</el-tag>
    <pre class="log">{{ log }}</pre>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from "vue";
import { api, type ToolboxTool } from "../api";
import { wsBaseUrl } from "../lib/wsLog";

const props = defineProps<{ toolId: string }>();

const tool = ref<ToolboxTool | null>(null);
const enabled = ref(true);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const values = reactive<Record<string, any>>({});
const log = ref("");
const status = ref("");
const running = computed(() => status.value === "running");
const statusType = computed(() =>
  status.value === "done" ? "success" : status.value === "failed" ? "danger" : "info",
);

let ws: WebSocket | null = null;
let offset = 0;

async function loadSnapshot() {
  const r = await api.toolboxLog(props.toolId, 0);
  log.value = r.chunk;
  offset = r.offset;
  status.value = r.status;
}

function openWs() {
  ws?.close();
  ws = new WebSocket(`${wsBaseUrl()}/api/v1/toolbox/tools/${encodeURIComponent(props.toolId)}/log/ws`);
  ws.onmessage = (ev) => {
    log.value += ev.data as string;
  };
  ws.onclose = async () => {
    await refreshStatus();
  };
}

async function refreshStatus() {
  status.value = (await api.toolboxRunStatus(props.toolId)).status;
}

async function run() {
  log.value = "";
  offset = 0;
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
  await loadSnapshot();
  if (status.value === "running") openWs();
});

onUnmounted(() => ws?.close());
</script>

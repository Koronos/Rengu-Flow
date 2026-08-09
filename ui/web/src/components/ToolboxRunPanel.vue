<!--
  The Toolbox page's run panel.

  The declared-input form and the console are no longer inline here: they live in
  `ToolboxInputsForm.vue` and `ToolboxLogPanel.vue` so the workflow `tool` node renders the exact
  same controls. What is left is what is genuinely page-specific — the head, the "execution
  disabled" banner and the Run/Cancel buttons.
-->
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

    <ToolboxInputsForm :values="values" :inputs="tool?.inputs || []" class="run-form" />

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

    <ToolboxLogPanel
      ref="logPanel"
      :tool-id="toolId"
      :last-run="tool?.last_run"
      @update:status="status = $event"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, useTemplateRef } from "vue";
import { CaretRight, Close } from "@element-plus/icons-vue";
import ToolboxInputsForm from "./ToolboxInputsForm.vue";
import ToolboxLogPanel from "./ToolboxLogPanel.vue";
import { api, type ToolboxTool } from "../api";

const props = defineProps<{ toolId: string }>();

const tool = ref<ToolboxTool | null>(null);
const enabled = ref(true);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const values = reactive<Record<string, any>>({});
/** Mirrored up from the log panel, which owns the stream that reports it. */
const status = ref("");
const running = computed(() => status.value === "running");
const statusType = computed(() =>
  status.value === "done" ? "success" : status.value === "failed" ? "danger" : "info",
);

const logPanel = useTemplateRef<InstanceType<typeof ToolboxLogPanel>>("logPanel");

async function run() {
  await api.runToolboxTool(props.toolId, { ...values });
  logPanel.value?.start();
}

async function cancel() {
  await api.cancelToolboxRun(props.toolId);
  await logPanel.value?.refreshStatus();
}

onMounted(async () => {
  tool.value = await api.getToolboxTool(props.toolId);
  enabled.value = (await api.toolboxEnabled()).enabled;
  for (const inp of tool.value.inputs) {
    if (inp.default !== undefined && inp.default !== null) values[inp.param] = inp.default;
  }
  if (tool.value.last_run?.inputs) Object.assign(values, tool.value.last_run.inputs);
  // The log panel takes it from here: `last_run` tells it whether to stream or load a snapshot.
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
</style>

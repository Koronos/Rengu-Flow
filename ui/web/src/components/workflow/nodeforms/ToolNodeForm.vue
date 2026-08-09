<!--
  The `tool` node's config: which Toolbox tool to run, and the values for the inputs it declares.

  The controls are `ToolboxInputsForm.vue` — the same block the `/toolbox` page renders — so a
  tool author who adds an input sees it appear in both places without a second edit here.

  Two things are specific to running a tool *as a node*:

  * **Injection by convention** (`workflow_nodes._build_tool_launch`): a tool that declares a
    `path` input gets the incoming folder written into it at launch, along with `caption_format`
    and `caption_ext` if it declares those too. Those params are therefore not editable here —
    showing a box whose value the executor overwrites is a lie — and a tool that declares no
    `path` is flagged as a pass-through, because that is exactly what it will be.
  * **`[toolbox].enabled`** gates execution app-wide. The node saves and edits either way, but it
    will not run, so the banner says so in the same words the Toolbox page uses.
-->
<template>
  <div class="tool-node-form">
    <el-alert
      v-if="toolboxEnabled === false"
      type="warning"
      class="tool-node-form__banner"
      :closable="false"
      show-icon
      title="Tool execution disabled"
      description="Set [toolbox].enabled = true in rengu.local.toml to run tools. Until then this step fails when the workflow reaches it; you can still edit and save."
    />

    <el-form label-position="top" :disabled="disabled">
      <el-form-item label="Tool" required>
        <el-select
          v-model="toolId"
          filterable
          :loading="listLoading"
          placeholder="Pick a tool from the Toolbox"
          class="w-full"
        >
          <el-option v-for="entry in tools" :key="entry.id" :label="entry.name" :value="entry.id">
            <span class="opt-title">{{ entry.name }}</span>
            <span class="opt-note">{{ entry.description }}</span>
          </el-option>
        </el-select>
        <el-text v-if="!listLoading && !tools.length" size="small" type="warning" class="hint-text">
          No tools in the Toolbox yet. Create one there first — this step has nothing to run.
        </el-text>
        <el-text v-else-if="loadError" size="small" type="danger" class="hint-text">
          {{ loadError }}
        </el-text>
      </el-form-item>
    </el-form>

    <template v-if="tool">
      <el-alert
        v-if="!declaresPath"
        type="info"
        :closable="false"
        show-icon
        class="tool-node-form__banner"
        title="This tool declares no 'path' input, so the incoming folder is never handed to it. The step runs and passes its input folder through unchanged."
      />
      <el-alert
        v-else
        type="info"
        :closable="false"
        show-icon
        class="tool-node-form__banner"
        :title="injectionNote"
      />

      <ToolboxInputsForm :values="values" :inputs="editableInputs" :disabled="disabled" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from "vue";
import ToolboxInputsForm from "../../ToolboxInputsForm.vue";
import { api, type ToolboxTool, type ToolboxToolSummary } from "../../../api";
import { formatError } from "../../../lib/formatError";

const config = defineModel<Record<string, unknown>>({ required: true });

defineProps({
  /** Read-only: the runner owns the workflow while it runs. */
  disabled: { type: Boolean, default: false },
});

/** Filled by the executor from the incoming handle; never edited on the node. */
const INJECTED_PARAMS = ["path", "caption_format", "caption_ext"];

const tools = ref<ToolboxToolSummary[]>([]);
const tool = ref<ToolboxTool | null>(null);
const listLoading = ref(false);
const toolboxEnabled = ref<boolean | null>(null);
const loadError = ref("");

const toolId = computed<string>({
  get: () => (typeof config.value.tool_id === "string" ? config.value.tool_id : ""),
  set: (value) => {
    config.value = { ...config.value, tool_id: value };
  },
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function configValues(source: Record<string, unknown>): Record<string, any> {
  const raw = source.values;
  return raw && typeof raw === "object" ? { ...(raw as Record<string, unknown>) } : {};
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const values = reactive<Record<string, any>>(configValues(config.value));

function sameAsValues(source: Record<string, unknown>): boolean {
  return JSON.stringify(configValues(source)) === JSON.stringify({ ...values });
}

// Mirror image of FolderNodeForm's pair: each watcher stops as soon as the other side agrees.
watch(
  () => config.value,
  (next) => {
    if (sameAsValues(next)) return;
    for (const key of Object.keys(values)) delete values[key];
    Object.assign(values, configValues(next));
  },
);

watch(
  values,
  () => {
    if (sameAsValues(config.value)) return;
    config.value = { ...config.value, values: { ...values } };
  },
  { deep: true },
);

const declaresPath = computed(() =>
  (tool.value?.inputs ?? []).some((input) => input.param === "path"),
);

const editableInputs = computed(() =>
  (tool.value?.inputs ?? []).filter((input) => !INJECTED_PARAMS.includes(input.param)),
);

const injectionNote = computed(() => {
  const injected = (tool.value?.inputs ?? [])
    .filter((input) => INJECTED_PARAMS.includes(input.param))
    .map((input) => input.param);
  return `The incoming folder is written into ${injected.join(", ")} when the step runs, so those inputs are not editable here.`;
});

async function loadTool(id: string): Promise<void> {
  if (!id) {
    tool.value = null;
    return;
  }
  try {
    const loaded = await api.getToolboxTool(id);
    tool.value = loaded;
    // Seed only what the user has not already set: a saved node keeps its own values.
    for (const input of loaded.inputs) {
      if (INJECTED_PARAMS.includes(input.param)) continue;
      if (values[input.param] === undefined && input.default !== undefined && input.default !== null) {
        values[input.param] = input.default;
      }
    }
  } catch (e) {
    tool.value = null;
    loadError.value = formatError(e);
  }
}

watch(toolId, (id) => void loadTool(id));

onMounted(async () => {
  listLoading.value = true;
  try {
    tools.value = await api.listToolboxTools();
  } catch (e) {
    loadError.value = formatError(e);
  } finally {
    listLoading.value = false;
  }
  try {
    toolboxEnabled.value = (await api.toolboxEnabled()).enabled;
  } catch {
    // The banner is advisory; a failed probe must not block editing the node.
  }
  await loadTool(toolId.value);
});
</script>

<style scoped>
.tool-node-form__banner {
  margin-bottom: 12px;
}
.hint-text {
  display: block;
  margin-top: 4px;
}
.opt-title {
  margin-right: 12px;
}
.opt-note {
  float: right;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>

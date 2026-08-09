<template>
  <el-popover
    v-model:visible="open"
    :width="288"
    :placement="placement"
    trigger="click"
    :disabled="disabled"
    popper-class="wf-add-popper"
  >
    <template #reference>
      <span class="wf-add__reference"><slot /></span>
    </template>

    <div class="wf-add">
      <section v-for="group in groups" :key="group.id" class="wf-add__group">
        <p class="wf-add__group-label">{{ group.label }}</p>
        <button
          v-for="entry in group.entries"
          :key="entry.key"
          type="button"
          class="wf-add__entry"
          @click="choose(entry)"
        >
          <span class="wf-add__entry-label">{{ entry.label }}</span>
          <span class="wf-add__entry-hint">{{ entry.hint }}</span>
        </button>
        <p v-if="!group.entries.length" class="wf-add__empty">{{ group.emptyHint }}</p>
      </section>
    </div>
  </el-popover>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { NODE_TYPE_GROUPS, describeOutput } from "../../lib/workflowNodeTypes";
import type { ToolboxToolSummary } from "../../api";

/**
 * The grouped add menu: Source · Prepare · Tools · Training.
 *
 * `Tools` is the only group that is not the static catalog — it is one entry per saved Toolbox
 * tool, because "add a tool step" without saying *which* tool would land an unconfigured node that
 * fails pre-flight on a missing `tool_id`. The tool list is fetched by the parent (one request per
 * editor, not one per popover) and handed in.
 */

interface AddEntry {
  key: string;
  label: string;
  hint: string;
  type: string;
  title: string;
  config?: Record<string, unknown>;
}

const props = withDefaults(
  defineProps<{
    tools?: ToolboxToolSummary[];
    disabled?: boolean;
    /** The end-of-chain button opens upward so the menu does not fall off the page. */
    placement?: "top" | "bottom";
  }>(),
  { tools: () => [], disabled: false, placement: "bottom" }
);

const emit = defineEmits<{
  select: [choice: { type: string; title: string; config?: Record<string, unknown> }];
}>();

const open = ref(false);

const groups = computed(() =>
  NODE_TYPE_GROUPS.map((group) => {
    if (group.id !== "tools") {
      return {
        id: group.id,
        label: group.label,
        emptyHint: "",
        entries: group.types.map(
          (spec): AddEntry => ({
            key: spec.type,
            label: spec.label,
            hint: describeOutput({ type: spec.type, config: {} }),
            type: spec.type,
            title: spec.label,
          })
        ),
      };
    }
    return {
      id: group.id,
      label: group.label,
      emptyHint: "No saved tools yet — create one in Toolbox.",
      entries: props.tools.map(
        (tool): AddEntry => ({
          key: `tool:${tool.id}`,
          label: tool.name || tool.id,
          hint: tool.description || "Toolbox tool",
          type: "tool",
          title: tool.name || tool.id,
          config: { tool_id: tool.id, values: {} },
        })
      ),
    };
  })
);

function choose(entry: AddEntry): void {
  open.value = false;
  emit("select", { type: entry.type, title: entry.title, config: entry.config });
}
</script>

<style scoped>
.wf-add__reference {
  display: inline-flex;
}

.wf-add {
  max-height: 60vh;
  overflow-y: auto;
  margin: -4px;
}

.wf-add__group + .wf-add__group {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--el-border-color-lighter);
}

.wf-add__group-label {
  margin: 0 0 4px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--el-text-color-secondary);
}

.wf-add__entry {
  display: block;
  width: 100%;
  text-align: left;
  border: none;
  background: none;
  font: inherit;
  cursor: pointer;
  padding: 5px 6px;
  border-radius: 6px;
  color: var(--el-text-color-primary);
}

.wf-add__entry:hover,
.wf-add__entry:focus-visible {
  background: var(--el-fill-color-light);
  outline: none;
}

.wf-add__entry-label {
  display: block;
  font-size: 13px;
}

.wf-add__entry-hint {
  display: block;
  font-size: 11px;
  line-height: 1.4;
  color: var(--el-text-color-secondary);
}

.wf-add__empty {
  margin: 0;
  padding: 4px 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>

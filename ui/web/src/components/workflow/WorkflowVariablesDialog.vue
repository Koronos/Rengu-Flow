<template>
  <el-dialog
    :model-value="open"
    title="Workflow variables"
    width="min(720px, 94vw)"
    @update:model-value="$emit('update:open', $event)"
  >
    <p class="wf-vars__intro">
      <code>${name}</code> in any text field — paths, model ids, prompts, output folders. Never
      numbers or booleans. Write <code>$$</code> for a literal <code>$</code>. Values are
      substituted once, at launch; a value that itself contains <code>${other}</code> stays literal.
    </p>

    <el-alert
      v-if="undefinedNames.length"
      type="error"
      show-icon
      :closable="false"
      class="mb-12"
      :title="`${undefinedNames.length} undefined ${undefinedNames.length === 1 ? 'variable' : 'variables'}: ${undefinedNames.join(', ')}`"
      description="The workflow cannot start until every referenced name has a value."
    />

    <KeyValueListField
      :model-value="asDict"
      hint="Name → value. Names match [A-Za-z_][A-Za-z0-9_]*."
      @update:model-value="onDictChange"
    />

    <h4 class="wf-vars__heading">Used by</h4>
    <el-table :data="usage" size="small" class="wf-vars__table" empty-text="No references yet">
      <el-table-column prop="name" label="Variable" width="180">
        <template #default="{ row }">
          <code>${{ '{' }}{{ row.name }}{{ '}' }}</code>
        </template>
      </el-table-column>
      <el-table-column label="Used by">
        <template #default="{ row }">
          <span v-if="!row.locations.length" class="wf-vars__unused">Not referenced</span>
          <span v-else class="wf-vars__locations">
            <span v-for="location in row.locations" :key="location" class="wf-vars__location">
              {{ location }}
            </span>
          </span>
        </template>
      </el-table-column>
      <el-table-column label="Defined" width="90">
        <template #default="{ row }">
          <span :class="row.defined ? 'wf-vars__ok' : 'wf-vars__missing'">
            {{ row.defined ? "✓" : "✕" }}
          </span>
        </template>
      </el-table-column>
    </el-table>

    <template #footer>
      <el-button @click="$emit('update:open', false)">Close</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed } from "vue";
import KeyValueListField from "../KeyValueListField.vue";
import { ordinalGlyph } from "../../lib/workflowGraph";
import { collectRefs } from "../../lib/workflowVars";
import { nodeTypeLabel } from "../../lib/workflowNodeTypes";
import type { WorkflowGraph, WorkflowVariable } from "../../types/workflow";

/**
 * Variables plus the column that makes them safe to edit: **who reads each one**.
 *
 * `collectRefs` reports every referenced name, including ones the workflow does not define — which
 * is exactly what the user needs to find, so undefined names are listed here (and counted in the
 * banner) rather than silently absent.
 *
 * Editing goes through `KeyValueListField`, which speaks `Record<string, string>`. A variable's
 * `description` is not in that shape, so it is carried across by name: renaming a variable drops
 * its description, keeping one preserves it.
 */

const props = defineProps<{
  open: boolean;
  graph: WorkflowGraph;
}>();

const emit = defineEmits<{
  "update:open": [value: boolean];
  "update:variables": [variables: WorkflowVariable[]];
}>();

const asDict = computed(() => {
  const out: Record<string, string> = {};
  for (const variable of props.graph.variables) out[variable.name] = variable.value;
  return out;
});

const refs = computed(() => collectRefs(props.graph));

/** `n2 · tag.models[0]` -> `② Tag · tag.models[0]`; an opaque uuid tells the reader nothing. */
const nodeLabels = computed(() => {
  const out: Record<string, string> = {};
  props.graph.nodes.forEach((node, index) => {
    out[node.id] = `${ordinalGlyph(index + 1)} ${node.title || nodeTypeLabel(node.type)}`;
  });
  return out;
});

function prettyLocation(location: string): string {
  const [nodeId, ...rest] = location.split(" · ");
  const label = nodeLabels.value[nodeId];
  return label ? [label, ...rest].join(" · ") : location;
}

const undefinedNames = computed(() => {
  const defined = new Set(props.graph.variables.map((variable) => variable.name));
  return Object.keys(refs.value).filter((name) => !defined.has(name));
});

const usage = computed(() => {
  const defined = props.graph.variables.map((variable) => variable.name);
  const names = [...new Set([...defined, ...Object.keys(refs.value)])].sort();
  return names.map((name) => ({
    name,
    defined: defined.includes(name),
    locations: (refs.value[name] ?? []).map(prettyLocation),
  }));
});

function onDictChange(value: unknown): void {
  const dict = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const descriptions = new Map(
    props.graph.variables.map((variable) => [variable.name, variable.description])
  );
  emit(
    "update:variables",
    Object.entries(dict).map(([name, raw]) => ({
      name,
      value: typeof raw === "string" ? raw : String(raw ?? ""),
      description: descriptions.get(name) ?? "",
    }))
  );
}
</script>

<style scoped>
.wf-vars__intro {
  margin: 0 0 12px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--el-text-color-secondary);
}

.wf-vars__heading {
  margin: 18px 0 8px;
  font-size: 13px;
}

.wf-vars__table {
  width: 100%;
}

.wf-vars__locations {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 10px;
}

.wf-vars__location {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.wf-vars__unused {
  font-size: 12px;
  color: var(--el-text-color-placeholder);
}

.wf-vars__ok {
  color: var(--el-color-success);
}

.wf-vars__missing {
  color: var(--el-color-danger);
}
</style>

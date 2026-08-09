<!--
  One control per declared tool input.

  Extracted verbatim from `ToolboxRunPanel.vue` so the Toolbox page and the workflow `tool` node
  drawer render a tool's inputs from the same markup — a control added here shows up in both.

  The root element is the `el-form` itself, so a consumer's class (`run-form`) still lands on it
  through attribute fallthrough and the page's layout is unchanged.
-->
<template>
  <el-form label-position="top" :disabled="disabled">
    <el-empty v-if="!inputs.length" description="No inputs" :image-size="48" />
    <el-form-item
      v-for="inp in inputs"
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
</template>

<script setup lang="ts">
import type { PropType } from "vue";
import type { ToolboxInput } from "../api";

defineProps({
  /**
   * The values object, keyed by `param`. The controls write **members** of the caller's own
   * reactive object — never the binding itself — which is exactly what the inline block did
   * before the extraction, so `ToolboxRunPanel`'s `reactive({})` keeps working untouched.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  values: { type: Object as PropType<Record<string, any>>, required: true },
  inputs: { type: Array as PropType<ToolboxInput[]>, default: () => [] },
  /**
   * Read-only. `el-form` hands this to every control under it, so one binding freezes the block —
   * the workflow drawer uses it while the runner owns the workflow.
   */
  disabled: { type: Boolean, default: false },
});
</script>

<style scoped>
.hint {
  display: block;
  margin-top: 4px;
  color: var(--el-text-color-secondary);
  font-size: 12px;
}
</style>

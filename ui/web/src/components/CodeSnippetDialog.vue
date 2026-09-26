<template>
  <el-dialog
    :model-value="modelValue"
    :title="title || 'Snippet'"
    width="min(720px, 96vw)"
    append-to-body
    class="code-snippet-dialog"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <pre class="code-snippet-pre"><code>{{ content }}</code></pre>
    <template #footer>
      <el-button @click="emit('update:modelValue', false)">Close</el-button>
      <el-button type="primary" @click="onCopy">Copy</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ElMessage } from "element-plus";
import { copyText } from "../lib/clipboard";

const props = defineProps<{
  modelValue: boolean;
  title?: string;
  content: string;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
}>();

async function onCopy(): Promise<void> {
  const ok = await copyText(props.content);
  if (ok) {
    ElMessage.success("Copied to clipboard");
  } else {
    ElMessage.error("Copy failed");
  }
}
</script>

<style scoped>
.code-snippet-pre {
  margin: 0;
  max-height: 60vh;
  overflow: auto;
  background: var(--el-fill-color-darker);
  border-radius: 8px;
  padding: 14px 16px;
}
.code-snippet-pre code {
  font-family: var(--rf-font-mono, ui-monospace, monospace);
  font-size: 13px;
  line-height: 1.5;
  color: var(--el-text-color-primary);
  white-space: pre;
}
</style>

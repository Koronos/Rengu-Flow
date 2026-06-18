<template>
  <div ref="host" class="code-editor"></div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { EditorView, basicSetup } from "codemirror";
import { EditorState } from "@codemirror/state";
import { indentUnit } from "@codemirror/language";
import { keymap, placeholder as cmPlaceholder } from "@codemirror/view";
import { indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";

const props = withDefaults(
  defineProps<{ modelValue: string; placeholder?: string }>(),
  { placeholder: "" },
);
const emit = defineEmits<{ "update:modelValue": [string] }>();

const host = ref<HTMLElement>();
let view: EditorView | null = null;

onMounted(() => {
  const onChange = EditorView.updateListener.of((u) => {
    if (!u.docChanged) return;
    const value = u.state.doc.toString();
    if (value !== props.modelValue) emit("update:modelValue", value);
  });
  view = new EditorView({
    parent: host.value,
    state: EditorState.create({
      doc: props.modelValue,
      extensions: [
        basicSetup,
        keymap.of([indentWithTab]),
        python(),
        oneDark,
        indentUnit.of("    "),
        EditorView.lineWrapping,
        cmPlaceholder(props.placeholder),
        onChange,
      ],
    }),
  });
});

// Reflect external changes (e.g. loading a tool) without breaking the cursor.
watch(
  () => props.modelValue,
  (value) => {
    if (view && value !== view.state.doc.toString()) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
    }
  },
);

onBeforeUnmount(() => {
  view?.destroy();
  view = null;
});
</script>

<style scoped>
.code-editor {
  font-size: 13px;
}
.code-editor :deep(.cm-editor) {
  min-height: 280px;
}
.code-editor :deep(.cm-editor.cm-focused) {
  outline: none;
}
.code-editor :deep(.cm-scroller) {
  max-height: 60vh;
  font-family: var(--rf-font-mono);
  line-height: 1.55;
}
.code-editor :deep(.cm-gutters) {
  border-right: 1px solid var(--el-border-color-lighter);
}
</style>

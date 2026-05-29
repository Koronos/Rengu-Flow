<template>
  <div class="editor-mode-wrap">
    <el-button-group size="small" class="editor-mode-toggle">
      <el-button
        :type="modelValue === 'form' ? 'primary' : 'default'"
        @click="setMode('form')"
      >
        Form
      </el-button>
      <el-button
        :type="modelValue === 'toml' ? 'primary' : 'default'"
        @click="setMode('toml')"
      >
        TOML
      </el-button>
    </el-button-group>
    <span class="editor-mode-hint muted-text">Form for guided edits; TOML for full file control.</span>
  </div>
</template>

<script setup lang="ts">
defineProps({
  modelValue: { type: String, default: "form" },
});

const emit = defineEmits(["update:modelValue"]);

function setMode(mode: "form" | "toml") {
  if (mode !== "form" && mode !== "toml") return;
  emit("update:modelValue", mode);
}
</script>

<style scoped>
.editor-mode-wrap {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}
.editor-mode-toggle {
  flex-shrink: 0;
}
.editor-mode-hint {
  font-size: 11px;
  line-height: 1.3;
  max-width: 220px;
  text-align: right;
}
</style>

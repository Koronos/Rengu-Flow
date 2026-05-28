<template>
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
.editor-mode-toggle {
  flex-shrink: 0;
}
</style>

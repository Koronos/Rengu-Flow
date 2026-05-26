<template>
  <el-button-group size="small" class="editor-mode-toggle">
    <el-button
      :type="model === 'form' ? 'primary' : 'default'"
      @click="model = 'form'"
    >
      Form
    </el-button>
    <el-button
      :type="model === 'toml' ? 'primary' : 'default'"
      @click="model = 'toml'"
    >
      TOML
    </el-button>
  </el-button-group>
</template>

<script setup>
const model = defineModel({ type: String, default: "form" });
</script>

<style scoped>
.editor-mode-toggle {
  flex-shrink: 0;
}
</style>

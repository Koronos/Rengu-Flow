<template>
  <el-radio-group :model-value="modelValue" size="small" class="dp-view-toggle" @change="onChange">
    <el-radio-button value="cards">
      <el-icon class="dp-view-toggle-icon"><Grid /></el-icon>
      <span>Cards</span>
    </el-radio-button>
    <el-radio-button value="list">
      <el-icon class="dp-view-toggle-icon"><List /></el-icon>
      <span>List</span>
    </el-radio-button>
    <el-radio-button value="table">
      <el-icon class="dp-view-toggle-icon"><Notebook /></el-icon>
      <span>Table</span>
    </el-radio-button>
  </el-radio-group>
</template>

<script setup>
import { Grid, List, Notebook } from "@element-plus/icons-vue";

defineProps({
  modelValue: { type: String, default: "cards" },
});

const emit = defineEmits(["update:modelValue"]);

function onChange(val) {
  emit("update:modelValue", val);
}
</script>

<style scoped>
.dp-view-toggle :deep(.el-radio-button__inner) {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
}
.dp-view-toggle-icon {
  font-size: 14px;
}
</style>

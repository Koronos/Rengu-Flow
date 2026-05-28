<template>
  <div v-if="show" class="path-validation-feedback">
    <el-text v-if="loading" type="info" size="small">Checking path…</el-text>
    <el-text v-else-if="error" type="danger" size="small">{{ error }}</el-text>
    <el-text v-else-if="ok" type="success" size="small">{{ okText }}</el-text>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = withDefaults(
  defineProps<{
    loading?: boolean;
    error?: string;
    ok?: boolean;
    okText?: string;
  }>(),
  {
    loading: false,
    error: "",
    ok: false,
    okText: "Path found",
  }
);

const show = computed(
  () => props.loading || !!props.error || props.ok
);
</script>

<style scoped>
.path-validation-feedback {
  margin-top: 4px;
  line-height: 1.4;
  overflow-wrap: anywhere;
  word-break: break-word;
}
</style>

<script setup lang="ts">
import DatasetViewModeToggle from "./DatasetViewModeToggle.vue";

const modelValue = defineModel<string>({ default: "cards" });
</script>

<template>
  <DatasetViewModeToggle v-model="modelValue" />
</template>

<template>
  <el-alert type="warning" :closable="false" show-icon class="mb-12 pick-banner">
    <template #title>Using a saved preset for a training job</template>
    <template v-if="variant === 'list'">
      Optional: open a saved preset below, edit if needed, validate, then click
      <strong>Use for training job</strong> in the editor. You can also create and run a
      config directly from the editor, or
      <el-button type="primary" link @click="$emit('cancel')">return to Runs</el-button>.
    </template>
    <template v-else>
      Validate if needed, then click <strong>Use for training job</strong>, or just
      <strong>Run now</strong> to create and run in one step. You can also
      <el-button type="primary" link @click="$emit('cancel')">return to Runs</el-button>.
    </template>
  </el-alert>
</template>

<script setup lang="ts">
defineProps({
  variant: {
    type: String as () => "list" | "editor",
    default: "list",
  },
});

defineEmits(["cancel"]);
</script>

<style scoped>
.pick-banner {
  line-height: 1.5;
}
</style>

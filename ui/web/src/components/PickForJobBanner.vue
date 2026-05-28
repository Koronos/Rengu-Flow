<template>
  <el-alert type="warning" :closable="false" show-icon class="mb-12 pick-banner">
    <template #title>Choosing config for a training job</template>
    <template v-if="variant === 'list'">
      Open a config below, edit if needed, validate, then click
      <strong>Use for training job</strong> in the editor. You can also
      <el-button type="primary" link @click="$emit('cancel')">return to Runs</el-button>
      without selecting.
    </template>
    <template v-else>
      Validate if needed, then click <strong>Use for training job</strong>. You can also
      <el-button type="primary" link @click="$emit('cancel')">return to Runs</el-button>
      without selecting.
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

<template>
  <div class="maintenance-view page-shell">
    <div class="page-head">
      <div class="page-head-text">
        <p class="page-subtitle">Local development tools (destructive actions)</p>
      </div>
    </div>

    <el-alert
      type="info"
      show-icon
      class="mb-12"
      title="Maintenance disabled"
      description="The maintenance API is permanently disabled in this build. Destructive operations (DB reset, submodule update, dependency install) have been removed."
    />
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.mb-12 {
  margin-bottom: 12px;
}
</style>

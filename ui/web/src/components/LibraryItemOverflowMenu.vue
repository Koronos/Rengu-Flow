<template>
  <el-tooltip content="More actions" :show-after="300">
    <el-dropdown trigger="click" @command="onCommand">
      <el-button
        class="library-overflow-trigger"
        size="small"
        circle
        :icon="MoreFilled"
        @click.stop
      />
      <template #dropdown>
        <slot />
        <el-dropdown-item command="duplicate" :disabled="loading">
          <span class="rf-dropdown-item-label">
            <el-icon><CopyDocument /></el-icon>
            <span>Duplicate</span>
          </span>
        </el-dropdown-item>
        <el-dropdown-item command="delete" :disabled="loading" divided>
          <span class="rf-dropdown-item-label rf-dropdown-item-label--danger">
            <el-icon><Delete /></el-icon>
            <span>Delete</span>
          </span>
        </el-dropdown-item>
      </template>
    </el-dropdown>
  </el-tooltip>
</template>

<script setup lang="ts">
import { CopyDocument, Delete, MoreFilled } from "@element-plus/icons-vue";

defineProps({
  loading: { type: Boolean, default: false },
});

const emit = defineEmits<{
  duplicate: [];
  delete: [];
}>();

function onCommand(command: string | number) {
  if (command === "duplicate") emit("duplicate");
  else if (command === "delete") emit("delete");
}
</script>

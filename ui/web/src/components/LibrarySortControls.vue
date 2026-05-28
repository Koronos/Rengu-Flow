<template>
  <div class="library-sort-controls">
    <el-select
      :model-value="sortField"
      class="library-sort-field"
      @update:model-value="$emit('update:sortField', $event)"
    >
      <el-option
        v-for="opt in fieldOptions"
        :key="opt.value"
        :label="opt.label"
        :value="opt.value"
      />
    </el-select>
    <el-tooltip :content="orderButtonLabel">
      <el-button
        circle
        size="small"
        class="library-sort-order"
        @click="$emit('toggle-order')"
      >
        <el-icon>
          <ArrowDown v-if="sortOrder === 'desc'" />
          <ArrowUp v-else />
        </el-icon>
      </el-button>
    </el-tooltip>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from "vue";
import { ArrowDown, ArrowUp } from "@element-plus/icons-vue";

defineProps({
  sortField: { type: String, required: true },
  sortOrder: { type: String, required: true },
  fieldOptions: {
    type: Array as PropType<{ value: string; label: string }[]>,
    required: true,
  },
  orderButtonLabel: { type: String, default: "" },
});

defineEmits(["update:sortField", "toggle-order"]);
</script>

<style scoped>
.library-sort-controls {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}
.library-sort-field {
  width: 112px;
}
.library-sort-order {
  flex-shrink: 0;
}
</style>

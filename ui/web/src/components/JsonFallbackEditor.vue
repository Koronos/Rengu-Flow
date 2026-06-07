<template>
  <div class="json-fallback-editor">
    <el-input
      :model-value="jsonText"
      type="textarea"
      :rows="rows"
      class="field-full"
      :placeholder="placeholder"
      @update:model-value="(v) => $emit('update:json', v)"
    />
    <el-text v-if="error" type="danger" size="small" class="json-error">
      {{ error }}
    </el-text>
    <el-button
      v-if="canReturn"
      size="small"
      link
      class="mt-8"
      @click="$emit('return')"
    >
      {{ backLabel }}
    </el-button>
  </div>
</template>

<script setup lang="ts">
import type { PropType } from "vue";

defineProps({
  jsonText: { type: String, default: "" },
  error: { type: String as PropType<string | null>, default: null },
  canReturn: { type: Boolean, default: false },
  rows: { type: Number, default: 6 },
  placeholder: { type: String, default: "" },
  backLabel: { type: String, default: "Back to table editor" },
});

defineEmits(["update:json", "return"]);
</script>

<style scoped>
.json-fallback-editor {
  width: 100%;
}
.field-full {
  width: 100%;
}
.json-error {
  display: block;
  margin-top: 6px;
}
.mt-8 {
  margin-top: 8px;
}
</style>

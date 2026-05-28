<template>
  <template v-if="helpText || docPath">
    <el-popover
      v-if="helpText"
      placement="right"
      :width="300"
      trigger="hover"
      :show-after="200"
    >
      <template #reference>
        <el-button
          class="help-btn"
          :icon="InfoFilled"
          circle
          size="small"
          text
          @click.stop="openDoc"
        />
      </template>
      <div class="help-body">
        <p class="help-text">{{ helpText }}</p>
        <el-button
          v-if="docPath"
          type="primary"
          link
          size="small"
          @click.stop="openDoc"
        >
          Read full documentation
        </el-button>
      </div>
    </el-popover>
    <el-button
      v-else
      class="help-btn"
      :icon="InfoFilled"
      circle
      size="small"
      text
      @click.stop="openDoc"
    />
    <DocMarkdownDrawer v-model="drawerOpen" :doc-path="docPath" />
  </template>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { InfoFilled } from "@element-plus/icons-vue";
import DocMarkdownDrawer from "./DocMarkdownDrawer.vue";
import type { SchemaField } from "../types/forms";

const props = defineProps<{ field: SchemaField }>();

const drawerOpen = ref(false);

const helpText = computed(
  () => props.field.help || ""
);

const docPath = computed(() => props.field.doc_path || "");

function openDoc() {
  if (docPath.value) drawerOpen.value = true;
}
</script>

<style scoped>
.help-btn {
  margin-left: 4px;
  vertical-align: middle;
  color: var(--el-color-info);
}
.help-body {
  font-size: 13px;
  line-height: 1.5;
}
.help-text {
  margin: 0 0 8px;
  white-space: pre-wrap;
}
</style>

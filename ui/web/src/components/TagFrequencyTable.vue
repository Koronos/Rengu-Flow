<template>
  <div class="tag-freq">
    <div class="tag-freq__toolbar">
      <el-input
        v-model="search"
        placeholder="Filter tags…"
        size="small"
        clearable
        :prefix-icon="Search"
      />
      <el-select :model-value="scope" size="small" class="tag-freq__scope" @change="emit('scope-change', $event)">
        <el-option label="Line 1" value="line1" />
        <el-option label="Tag lines" value="tag_lines" />
        <el-option label="All lines" value="all_lines" />
      </el-select>
    </div>
    <div class="tag-freq__prune">
      <el-input-number v-model="pruneBelow" :min="2" size="small" controls-position="right" />
      <el-button size="small" @click="emit('prune', pruneBelow)">
        Prune tags seen &lt; n times
      </el-button>
    </div>
    <el-table
      :data="visibleTags"
      size="small"
      height="100%"
      class="tag-freq__table"
      @row-click="(row: TagRow) => emit('select-tag', row.tag)"
    >
      <el-table-column prop="tag" label="Tag" min-width="160" show-overflow-tooltip />
      <el-table-column prop="count" label="#" width="64" sortable align="right" />
      <el-table-column width="92" align="right">
        <template #default="{ row }">
          <span title="Rename tag">
            <el-button size="small" text :icon="EditPen" @click.stop="startRename(row.tag)" />
          </span>
          <span title="Remove tag from all images">
            <el-button
              size="small"
              text
              type="danger"
              :icon="Delete"
              @click.stop="emit('remove-tag', row.tag)"
            />
          </span>
        </template>
      </el-table-column>
    </el-table>
    <el-dialog v-model="renameOpen" title="Rename tag" width="360px" append-to-body>
      <el-form label-position="top">
        <el-form-item :label="`Rename '${renameFrom}' to:`">
          <el-input v-model="renameTo" autofocus @keydown.enter.prevent="confirmRename" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="renameOpen = false">Cancel</el-button>
        <el-button type="primary" :disabled="!renameTo.trim()" @click="confirmRename">
          Rename
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { Delete, EditPen, Search } from "@element-plus/icons-vue";

interface TagRow {
  tag: string;
  count: number;
}

const props = defineProps<{
  tags: TagRow[];
  scope: string;
}>();

const emit = defineEmits<{
  (e: "select-tag", tag: string): void;
  (e: "remove-tag", tag: string): void;
  (e: "rename-tag", from: string, to: string): void;
  (e: "prune", minCount: number): void;
  (e: "scope-change", scope: string): void;
}>();

const search = ref("");
const pruneBelow = ref(3);
const renameOpen = ref(false);
const renameFrom = ref("");
const renameTo = ref("");

const visibleTags = computed(() => {
  const needle = search.value.trim().toLowerCase();
  if (!needle) return props.tags;
  return props.tags.filter((t) => t.tag.toLowerCase().includes(needle));
});

function startRename(tag: string): void {
  renameFrom.value = tag;
  renameTo.value = tag;
  renameOpen.value = true;
}

function confirmRename(): void {
  const to = renameTo.value.trim();
  if (!to || to === renameFrom.value) {
    renameOpen.value = false;
    return;
  }
  emit("rename-tag", renameFrom.value, to);
  renameOpen.value = false;
}
</script>

<style scoped>
.tag-freq {
  display: flex;
  flex-direction: column;
  gap: 8px;
  height: 100%;
  min-height: 0;
}
.tag-freq__toolbar {
  display: flex;
  gap: 8px;
}
.tag-freq__scope {
  width: 110px;
  flex-shrink: 0;
}
.tag-freq__prune {
  display: flex;
  gap: 8px;
  align-items: center;
}
.tag-freq__table {
  flex: 1;
  min-height: 0;
  cursor: pointer;
}
</style>

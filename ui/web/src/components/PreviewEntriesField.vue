<template>
  <div class="preview-entries-field">
    <div class="entries-toolbar">
      <el-input
        v-model="query"
        clearable
        placeholder="Search sampling prompts by tag or prompt…"
        class="entries-search"
        :prefix-icon="Search"
      />
      <el-button type="primary" :icon="Plus" @click="openAdd">Add sampling</el-button>
    </div>

    <p v-if="entries.length" class="entries-count">
      <template v-if="filtered.length !== entries.length">
        {{ filtered.length }} of {{ entries.length }} configurations
      </template>
      <template v-else>
        {{ entries.length }}
        {{ entries.length === 1 ? "configuration" : "configurations" }}
      </template>
    </p>

    <el-table
      v-if="filtered.length"
      :data="filtered"
      stripe
      class="entries-table"
      @row-click="(row: Row) => openEdit(row.index)"
    >
      <el-table-column label="Tag / title" min-width="140">
        <template #default="{ row }">
          <span class="entry-title">{{ row.title }}</span>
        </template>
      </el-table-column>
      <el-table-column label="Prompt" min-width="200" show-overflow-tooltip>
        <template #default="{ row }">
          {{ row.subtitle }}
        </template>
      </el-table-column>
      <el-table-column label="" width="160" align="right">
        <template #default="{ row }">
          <div class="row-actions" @click.stop>
            <el-tag v-if="row.overrideCount" size="small" type="info">
              {{ row.overrideCount }} override{{ row.overrideCount === 1 ? "" : "s" }}
            </el-tag>
            <template v-if="!isMobile">
              <el-tooltip content="Edit" :show-after="300">
                <el-button size="small" circle :icon="Edit" @click="openEdit(row.index)" />
              </el-tooltip>
              <el-tooltip content="Duplicate" :show-after="300">
                <el-button size="small" circle :icon="CopyDocument" @click="duplicateAt(row.index)" />
              </el-tooltip>
              <el-tooltip content="Remove" :show-after="300">
                <el-button size="small" circle :icon="Delete" @click="removeAt(row.index)" />
              </el-tooltip>
            </template>
            <el-dropdown v-else trigger="click">
              <el-button size="small" circle :icon="MoreFilled" />
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item :icon="Edit" @click="openEdit(row.index)">Edit</el-dropdown-item>
                  <el-dropdown-item :icon="CopyDocument" @click="duplicateAt(row.index)">
                    Duplicate
                  </el-dropdown-item>
                  <el-dropdown-item :icon="Delete" @click="removeAt(row.index)">Remove</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </template>
      </el-table-column>
    </el-table>

    <el-empty
      v-else-if="entries.length && query.trim()"
      description="No sampling prompts match your search"
      :image-size="56"
    />
    <el-empty v-else description="No sampling prompts yet" :image-size="56">
      <el-button type="primary" :icon="Plus" @click="openAdd">Add sampling</el-button>
    </el-empty>

    <PreviewEntryDialog
      v-model="dialogOpen"
      :entry-fields="entryFields"
      :entry="dialogEntry"
      :edit-index="dialogIndex"
      :parent-form="parentForm"
      :capabilities="capabilities"
      @save="onDialogSave"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, type PropType } from "vue";
import { ElMessageBox } from "element-plus";
import { CopyDocument, Delete, Edit, MoreFilled, Plus, Search } from "@element-plus/icons-vue";
import PreviewEntryDialog from "./PreviewEntryDialog.vue";
import { useBreakpoint } from "../composables/useBreakpoint";
import {
  clonePreviewEntry,
  countPreviewEntryOverrides,
  duplicatePreviewEntry,
  normalizePreviewEntries,
  previewEntryName,
  previewEntrySubtitle,
  type PreviewEntry,
} from "../lib/previewEntries";
import type { FormValues, ModelCapabilities, SchemaField } from "../types/forms";

interface Row {
  index: number;
  title: string;
  subtitle: string;
  overrideCount: number;
}

const props = defineProps({
  modelValue: { type: [Array, String] as PropType<unknown> },
  entryFields: { type: Array as PropType<SchemaField[]>, default: () => [] },
  parentForm: { type: Object as PropType<FormValues>, default: () => ({}) },
  capabilities: { type: Object as PropType<ModelCapabilities>, default: () => ({}) },
});

const emit = defineEmits<{
  "update:modelValue": [value: PreviewEntry[]];
}>();

const { isMobile } = useBreakpoint();
const query = ref("");
const dialogOpen = ref(false);
const dialogIndex = ref(-1);
const dialogEntry = ref<PreviewEntry | null>(null);

const entries = computed(() => normalizePreviewEntries(props.modelValue));

const filtered = computed((): Row[] => {
  const q = query.value.trim().toLowerCase();
  return entries.value
    .map((entry, index) => ({
      index,
      title: previewEntryName(entry, index),
      subtitle: previewEntrySubtitle(entry, index),
      overrideCount: countPreviewEntryOverrides(entry),
    }))
    .filter((row) => {
      if (!q) return true;
      return (
        row.title.toLowerCase().includes(q) ||
        row.subtitle.toLowerCase().includes(q)
      );
    });
});

function emitEntries(next: PreviewEntry[]): void {
  emit("update:modelValue", next);
}

function openAdd(): void {
  dialogIndex.value = -1;
  dialogEntry.value = null;
  dialogOpen.value = true;
}

function openEdit(index: number): void {
  dialogIndex.value = index;
  dialogEntry.value = clonePreviewEntry(entries.value[index]);
  dialogOpen.value = true;
}

function onDialogSave({ entry, index }: { entry: PreviewEntry; index: number }): void {
  const next = [...entries.value];
  if (index >= 0) {
    next[index] = entry;
  } else {
    next.push(entry);
  }
  emitEntries(next);
}

function duplicateAt(index: number): void {
  const next = [...entries.value];
  next.splice(index + 1, 0, duplicatePreviewEntry(entries.value[index]));
  emitEntries(next);
}

async function removeAt(index: number): Promise<void> {
  const label = previewEntryName(entries.value[index], index);
  try {
    await ElMessageBox.confirm(`Remove preview "${label}"?`, "Remove preview", {
      type: "warning",
      confirmButtonText: "Remove",
    });
  } catch {
    return;
  }
  emitEntries(entries.value.filter((_, i) => i !== index));
}
</script>

<style scoped>
.preview-entries-field {
  width: 100%;
  margin-bottom: 16px;
}
.entries-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.entries-search {
  flex: 1;
  min-width: 160px;
}
.entries-count {
  margin: 0 0 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.entries-table {
  width: 100%;
  cursor: pointer;
}
.entry-title {
  font-weight: 500;
}
.row-actions {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
</style>

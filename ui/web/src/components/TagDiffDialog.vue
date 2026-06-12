<template>
  <el-dialog
    :model-value="modelValue"
    title="Review staged changes"
    width="720px"
    append-to-body
    @update:model-value="emit('update:modelValue', $event)"
  >
    <div v-if="loading" class="tag-diff__loading">
      <el-skeleton :rows="4" animated />
    </div>
    <template v-else-if="diff">
      <el-alert
        type="warning"
        :closable="false"
        show-icon
        class="tag-diff__alert"
      >
        {{ diff.total }} file(s) will change. A full backup of every caption file is
        taken before anything is written.
      </el-alert>
      <div class="tag-diff__list">
        <div v-for="entry in diff.entries" :key="entry.key" class="tag-diff__entry">
          <div class="tag-diff__key">
            {{ entry.key }}
            <el-tag v-if="entry.after === null" size="small" type="danger" effect="plain">
              quarantine
            </el-tag>
          </div>
          <template v-if="entry.after !== null">
            <div
              v-for="(line, i) in lineDiff(entry)"
              :key="i"
              class="tag-diff__line"
              :class="{
                'tag-diff__line--removed': line.kind === 'removed',
                'tag-diff__line--added': line.kind === 'added',
              }"
            >
              <span class="tag-diff__line-mark">{{
                line.kind === "removed" ? "−" : line.kind === "added" ? "+" : " "
              }}</span>
              <span class="tag-diff__line-text">{{ line.text }}</span>
            </div>
          </template>
        </div>
        <div v-if="diff.total > diff.entries.length" class="tag-diff__more">
          … and {{ diff.total - diff.entries.length }} more file(s)
        </div>
      </div>
    </template>
    <template #footer>
      <el-button @click="emit('update:modelValue', false)">Cancel</el-button>
      <el-button type="primary" :loading="committing" @click="emit('commit')">
        Commit (with backup)
      </el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import type { TagDiffEntry, TagDiffResult } from "../types/api";

defineProps<{
  modelValue: boolean;
  diff: TagDiffResult | null;
  loading: boolean;
  committing: boolean;
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: boolean): void;
  (e: "commit"): void;
}>();

interface DiffLine {
  kind: "kept" | "removed" | "added";
  text: string;
}

function lineDiff(entry: TagDiffEntry): DiffLine[] {
  const before = entry.before ?? [];
  const after = entry.after ?? [];
  const afterSet = new Set(after);
  const beforeSet = new Set(before);
  const lines: DiffLine[] = [];
  for (const line of before) {
    lines.push({ kind: afterSet.has(line) ? "kept" : "removed", text: line });
  }
  for (const line of after) {
    if (!beforeSet.has(line)) lines.push({ kind: "added", text: line });
  }
  return lines;
}
</script>

<style scoped>
.tag-diff__alert {
  margin-bottom: 12px;
}
.tag-diff__list {
  max-height: 50vh;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.tag-diff__entry {
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  padding: 8px 10px;
}
.tag-diff__key {
  font-weight: 600;
  font-size: 13px;
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
  align-items: center;
}
.tag-diff__line {
  font-family: var(--el-font-family-mono, monospace);
  font-size: 12px;
  display: flex;
  gap: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}
.tag-diff__line--removed {
  background: var(--el-color-danger-light-9);
  color: var(--el-color-danger);
}
.tag-diff__line--added {
  background: var(--el-color-success-light-9);
  color: var(--el-color-success);
}
.tag-diff__line-mark {
  width: 12px;
  flex-shrink: 0;
  text-align: center;
}
.tag-diff__more {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>

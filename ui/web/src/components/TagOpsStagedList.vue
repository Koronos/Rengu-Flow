<template>
  <div class="staged-ops">
    <div class="staged-ops__header">
      <span class="staged-ops__title">
        Staged edits
        <el-tag v-if="ops.length" size="small" type="warning" effect="plain">
          {{ ops.length }}
        </el-tag>
      </span>
      <el-button
        size="small"
        :icon="RefreshLeft"
        :disabled="!ops.length"
        @click="emit('undo')"
      >
        Undo
      </el-button>
    </div>
    <el-empty
      v-if="!ops.length"
      description="No staged edits — nothing is written until you commit"
      :image-size="48"
    />
    <ol v-else class="staged-ops__list">
      <li v-for="(op, i) in ops" :key="i" class="staged-ops__item">
        <el-tag size="small" :type="opTagType(op.op)" effect="plain">{{ op.op }}</el-tag>
        <span class="staged-ops__desc">{{ describe(op) }}</span>
      </li>
    </ol>
  </div>
</template>

<script setup lang="ts">
import { RefreshLeft } from "@element-plus/icons-vue";
import type { TagEditOpDto } from "../types/api";

defineProps<{ ops: TagEditOpDto[] }>();

const emit = defineEmits<{ (e: "undo"): void }>();

const SCOPE_LABELS: Record<string, string> = {
  line1: "line 1",
  tag_lines: "tag lines",
  all_lines: "all lines",
  line_n: "line N",
};

function opTagType(op: string): "primary" | "danger" | "warning" | "info" {
  if (op === "remove" || op === "quarantine") return "danger";
  if (op === "prune") return "warning";
  if (op === "rename") return "info";
  return "primary";
}

function filterText(filter: TagEditOpDto["filter"]): string {
  const parts: string[] = [];
  if (filter?.all?.length) parts.push(`all of [${filter.all.join(", ")}]`);
  if (filter?.any?.length) parts.push(`any of [${filter.any.join(", ")}]`);
  if (filter?.none?.length) parts.push(`lacking [${filter.none.join(", ")}]`);
  return parts.length ? ` where ${parts.join(" and ")}` : "";
}

function describe(op: TagEditOpDto): string {
  const scope = SCOPE_LABELS[op.scope ?? "tag_lines"] ?? op.scope;
  switch (op.op) {
    case "add":
      return `${(op.tags ?? []).join(", ")} → ${scope}${filterText(op.filter)}`;
    case "remove":
      return `${(op.tags ?? []).join(", ")} from ${scope}${filterText(op.filter)}`;
    case "rename":
      return `${(op.tags ?? []).join(", ")} → '${op.rename_to}' on ${scope}`;
    case "prune":
      return `tags seen < ${op.min_count} times on ${scope}`;
    case "quarantine":
      return `images${filterText(op.filter)}`;
    default:
      return JSON.stringify(op);
  }
}
</script>

<style scoped>
.staged-ops {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.staged-ops__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.staged-ops__title {
  font-size: 13px;
  font-weight: 600;
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.staged-ops__list {
  margin: 0;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 180px;
  overflow: auto;
}
.staged-ops__item {
  display: flex;
  gap: 6px;
  align-items: baseline;
  font-size: 12px;
}
.staged-ops__desc {
  color: var(--el-text-color-regular);
  word-break: break-word;
}
</style>

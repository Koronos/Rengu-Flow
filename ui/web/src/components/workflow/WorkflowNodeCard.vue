<template>
  <div class="wf-node">
    <div
      class="wf-card"
      :class="{
        'wf-card--active': active,
        'wf-card--stale': stale,
        'wf-card--disabled': !node.enabled,
        'wf-card--dimmed': dimmed,
        'wf-card--source': highlight === 'source',
        'wf-card--target': highlight === 'target',
      }"
      role="button"
      tabindex="0"
      @click="$emit('open')"
      @keydown.enter.prevent="$emit('open')"
      @keydown.space.prevent="$emit('open')"
      @mouseenter="$emit('card-enter')"
      @mouseleave="$emit('card-leave')"
    >
      <div class="wf-card__head">
        <span class="wf-card__ordinal" aria-hidden="true">{{ ordinalGlyph(position) }}</span>
        <el-icon class="wf-card__icon"><component :is="icon" /></el-icon>
        <span class="wf-card__title">{{ node.title || nodeTypeLabel(node.type) }}</span>

        <!--
          The badge is drawn ONLY for a jump. When `from` is the card right above, the connector
          already says it and a badge there would be noise on every card — at which point no badge
          is read anywhere. It is a <button> so the highlight works from the keyboard too.
        -->
        <button
          v-if="jump"
          type="button"
          class="wf-card__badge"
          :title="`Reads from step ${jump.sourcePosition}. Enter to jump to it.`"
          @click.stop="$emit('badge-activate')"
          @keydown.enter.stop.prevent="$emit('badge-activate')"
          @mouseenter="$emit('badge-enter')"
          @mouseleave="$emit('badge-leave')"
          @focus="$emit('badge-enter')"
          @blur="$emit('badge-leave')"
        >
          ⟵ from {{ ordinalGlyph(jump.sourcePosition) }}
        </button>

        <span class="wf-card__spacer" />

        <span v-if="stale" class="wf-card__stale" title="Configuration changed since this ran">
          ◌ stale
        </span>
        <span class="wf-card__chip" :class="`wf-card__chip--${chip.tone}`">
          <span aria-hidden="true">{{ chip.glyph }}</span>
          <span>{{ chip.label }}</span>
          <span v-if="chip.showProgress && percent != null" class="wf-card__percent">
            {{ percent }}%
          </span>
        </span>

        <el-dropdown trigger="click" @command="onCommand">
          <el-button
            class="wf-card__menu"
            size="small"
            circle
            :icon="MoreFilled"
            v-bind="ariaLabel(`Actions for step ${position}`)"
            @click.stop
          />
          <template #dropdown>
            <el-dropdown-item command="open">
              <span class="rf-dropdown-item-label">
                <el-icon><Edit /></el-icon><span>Open</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="run-from" :disabled="runDisabled" divided>
              <span class="rf-dropdown-item-label">
                <el-icon><CaretRight /></el-icon>
                <span>{{ runFromReason || "Run from here" }}</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="run-only" :disabled="runDisabled">
              <span class="rf-dropdown-item-label">
                <el-icon><CaretRight /></el-icon><span>Run only this step</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="up" :disabled="readOnly || !!upReason" divided>
              <span class="rf-dropdown-item-label">
                <el-icon><Top /></el-icon>
                <span>{{ upReason || "Move up" }}</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="down" :disabled="readOnly || !!downReason">
              <span class="rf-dropdown-item-label">
                <el-icon><Bottom /></el-icon>
                <span>{{ downReason || "Move down" }}</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="toggle" :disabled="readOnly" divided>
              <span class="rf-dropdown-item-label">
                <el-icon><Switch /></el-icon>
                <span>{{ node.enabled ? "Disable step" : "Enable step" }}</span>
              </span>
            </el-dropdown-item>
            <el-dropdown-item command="delete" :disabled="readOnly">
              <span class="rf-dropdown-item-label rf-dropdown-item-label--danger">
                <el-icon><Delete /></el-icon><span>Delete</span>
              </span>
            </el-dropdown-item>
          </template>
        </el-dropdown>
      </div>

      <p class="wf-card__summary">{{ summary }}</p>
      <p class="wf-card__output">→ {{ outputSentence }}</p>

      <el-progress
        v-if="chip.showProgress"
        class="wf-card__progress"
        :percentage="percent ?? 0"
        :stroke-width="4"
        :show-text="false"
        :indeterminate="percent == null"
      />

      <p v-if="chip.detail" class="wf-card__detail" :class="`wf-card__detail--${chip.tone}`">
        {{ chip.detail }}
      </p>
    </div>

    <!--
      The "reads past this step" legend, shown on the SKIPPED card while a jump is hovered. Without
      it a link that leaps over a step looks like a mistake in the chain rather than a choice.
    -->
    <p v-if="skippedNote" class="wf-card__legend">{{ skippedNote }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import {
  Bottom,
  CaretRight,
  ChatLineSquare,
  DataAnalysis,
  Delete,
  Edit,
  Filter,
  Folder,
  MoreFilled,
  PriceTag,
  QuestionFilled,
  Scissor,
  Switch,
  Tools,
  Top,
  VideoPlay,
} from "@element-plus/icons-vue";
import type { Component } from "vue";
import { ariaLabel } from "../../lib/aria";
import { ordinalGlyph } from "../../lib/workflowGraph";
import { nodeTypeIcon, nodeTypeLabel } from "../../lib/workflowNodeTypes";
import type { NodeChip } from "../../lib/workflowStatus";
import type { WorkflowNode } from "../../types/workflow";

/**
 * One step in the vertical chain.
 *
 * The card owns no state and derives nothing: the chip, the summary, the jump and the highlight
 * all arrive as props, because whether a card is dimmed depends on what the user is hovering
 * *elsewhere*. Keeping that in the view is what lets one hover repaint the whole chain coherently.
 */

const props = withDefaults(
  defineProps<{
    node: WorkflowNode;
    /** 1-based position — the ① ② ③ everything else refers to. */
    position: number;
    chip: NodeChip;
    summary: string;
    outputSentence: string;
    stale?: boolean;
    /** 0-100 from the node log's progress marker; `null` while a running node has not reported. */
    percent?: number | null;
    /** Set only when `from` skips at least one card. */
    jump?: { key: string; sourcePosition: number } | null;
    highlight?: "source" | "target" | "none";
    dimmed?: boolean;
    /** e.g. "③ reads past this step" — shown under a card a hovered jump reads past. */
    skippedNote?: string;
    active?: boolean;
    readOnly?: boolean;
    /** Non-empty when *Run from here* is blocked; doubles as the menu label so the reason is read. */
    runFromReason?: string;
    upReason?: string;
    downReason?: string;
  }>(),
  {
    stale: false,
    percent: null,
    jump: null,
    highlight: "none",
    dimmed: false,
    skippedNote: "",
    active: false,
    readOnly: false,
    runFromReason: "",
    upReason: "",
    downReason: "",
  }
);

const emit = defineEmits<{
  open: [];
  "card-enter": [];
  "card-leave": [];
  "badge-enter": [];
  "badge-leave": [];
  "badge-activate": [];
  "run-from": [];
  "run-only": [];
  move: [direction: "up" | "down"];
  "toggle-enabled": [];
  delete: [];
}>();

const ICONS: Record<string, Component> = {
  Folder,
  PriceTag,
  ChatLineSquare,
  Scissor,
  Filter,
  DataAnalysis,
  Tools,
  VideoPlay,
  QuestionFilled,
};

const icon = computed(() => ICONS[nodeTypeIcon(props.node.type)] ?? QuestionFilled);

const percent = computed(() => {
  if (props.percent == null) return null;
  return Math.max(0, Math.min(100, Math.round(props.percent)));
});

const runDisabled = computed(() => props.readOnly || !props.node.enabled || !!props.runFromReason);

function onCommand(command: string | number): void {
  switch (command) {
    case "open":
      emit("open");
      break;
    case "run-from":
      emit("run-from");
      break;
    case "run-only":
      emit("run-only");
      break;
    case "up":
      emit("move", "up");
      break;
    case "down":
      emit("move", "down");
      break;
    case "toggle":
      emit("toggle-enabled");
      break;
    case "delete":
      emit("delete");
      break;
    default:
      break;
  }
}
</script>

<style scoped>
.wf-node {
  min-width: 0;
}

.wf-card {
  border: 1px solid var(--el-border-color);
  border-radius: 8px;
  background: var(--el-bg-color-overlay);
  padding: 10px 12px 12px;
  cursor: pointer;
  transition: border-color 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
}

.wf-card:hover,
.wf-card:focus-visible {
  border-color: var(--el-color-primary);
  outline: none;
}

.wf-card--active {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 1px var(--el-color-primary) inset;
}

/* done AND stale at once is information, not a contradiction: the ring is drawn over any chip. */
.wf-card--stale {
  border-style: dashed;
  border-color: var(--el-color-warning);
}

.wf-card--disabled {
  opacity: 0.6;
}

.wf-card--disabled .wf-card__title {
  text-decoration: line-through;
}

.wf-card--dimmed {
  opacity: 0.35;
}

.wf-card--source {
  border-color: var(--el-color-primary);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--el-color-primary) 35%, transparent);
}

.wf-card--target {
  border-color: var(--el-color-primary);
}

.wf-card__head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
}

.wf-card__ordinal {
  font-size: 15px;
  color: var(--el-text-color-secondary);
  flex: 0 0 auto;
}

.wf-card__icon {
  color: var(--el-text-color-secondary);
  flex: 0 0 auto;
}

.wf-card__title {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.wf-card__spacer {
  flex: 1 1 auto;
}

.wf-card__badge {
  flex: 0 0 auto;
  font: inherit;
  font-size: 12px;
  line-height: 1;
  padding: 3px 7px;
  border-radius: 10px;
  cursor: pointer;
  color: var(--el-color-primary);
  background: color-mix(in srgb, var(--el-color-primary) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--el-color-primary) 40%, transparent);
}

.wf-card__badge:hover,
.wf-card__badge:focus-visible {
  background: color-mix(in srgb, var(--el-color-primary) 22%, transparent);
  outline: none;
}

.wf-card__stale {
  flex: 0 0 auto;
  font-size: 12px;
  color: var(--el-color-warning);
}

.wf-card__chip {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  white-space: nowrap;
  color: var(--el-text-color-secondary);
}

.wf-card__chip--warning {
  color: var(--el-color-warning);
}
.wf-card__chip--primary {
  color: var(--el-color-primary);
}
.wf-card__chip--success {
  color: var(--el-color-success);
}
.wf-card__chip--danger {
  color: var(--el-color-danger);
}

.wf-card__percent {
  font-variant-numeric: tabular-nums;
}

.wf-card__menu {
  flex: 0 0 auto;
}

.wf-card__summary,
.wf-card__output,
.wf-card__detail {
  margin: 4px 0 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--el-text-color-secondary);
  word-break: break-word;
}

.wf-card__output {
  color: var(--el-text-color-placeholder);
}

.wf-card__detail--danger {
  color: var(--el-color-danger);
}
.wf-card__detail--warning {
  color: var(--el-color-warning);
}

.wf-card__progress {
  margin-top: 8px;
}

.wf-card__legend {
  margin: 4px 0 0 4px;
  font-size: 11px;
  color: var(--el-color-primary);
  opacity: 0.85;
}
</style>

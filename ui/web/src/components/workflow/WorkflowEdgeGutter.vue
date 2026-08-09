<template>
  <div class="wf-gutter" :class="{ 'wf-gutter--dimmed': dimmed }">
    <template v-for="segment in segments" :key="`${segment.key}:${segment.span}`">
      <span
        class="wf-gutter__line"
        :class="[`wf-gutter__line--${segment.span}`, lineClass(segment)]"
        :style="lineStyle(segment)"
      />
      <span
        v-if="segment.cap"
        class="wf-gutter__cap"
        :class="lineClass(segment)"
        :style="capStyle(segment)"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import type { GutterSegment } from "../../lib/workflowGutter";

/**
 * One row's slice of the connector rail.
 *
 * Every line is a plain absolutely-positioned `<span>` — no canvas, no SVG, no measurement. A row
 * paints only the piece of each edge that crosses it, so a card that grows (a long error, a wide
 * summary) moves its own piece and nothing else: the rail can never drift out of step with the
 * cards it connects.
 */

const props = withDefaults(
  defineProps<{
    segments?: GutterSegment[];
    /** Edge keys to paint in the accent colour — the hovered card's links, or one hovered badge. */
    activeKeys?: string[];
    /** Something elsewhere has focus; everything here fades back. */
    dimmed?: boolean;
  }>(),
  { segments: () => [], activeKeys: () => [], dimmed: false }
);

/** Lane 0 sits closest to the card; each jump lane steps another notch left. */
const LANE_BASE_PX = 6;
const LANE_GAP_PX = 8;
/** Where a connector meets its card, measured from the row's top — level with the card's title. */
const ANCHOR_PX = 26;

function laneRight(lane: number): number {
  return LANE_BASE_PX + Math.max(0, lane) * LANE_GAP_PX;
}

function lineClass(segment: GutterSegment): string {
  if (props.activeKeys.includes(segment.key)) return "is-active";
  return segment.lane > 0 ? "is-jump" : "";
}

function lineStyle(segment: GutterSegment): Record<string, string> {
  const right = `${laneRight(segment.lane)}px`;
  if (segment.span === "top") return { right, top: "0", height: `${ANCHOR_PX}px` };
  if (segment.span === "bottom") return { right, top: `${ANCHOR_PX}px`, bottom: "0" };
  return { right, top: "0", bottom: "0" };
}

function capStyle(segment: GutterSegment): Record<string, string> {
  return { top: `${ANCHOR_PX}px`, right: "0", width: `${laneRight(segment.lane)}px` };
}
</script>

<style scoped>
.wf-gutter {
  position: relative;
  width: 28px;
  align-self: stretch;
  flex: 0 0 28px;
}

.wf-gutter__line,
.wf-gutter__cap {
  position: absolute;
  background: var(--el-border-color);
  transition: background-color 0.15s ease, opacity 0.15s ease;
  pointer-events: none;
}

.wf-gutter__line {
  width: 2px;
  border-radius: 1px;
}

.wf-gutter__cap {
  height: 2px;
  border-radius: 1px;
}

/* A jump already reads as an exception through the badge; the rail only needs to agree. */
.wf-gutter__line.is-jump,
.wf-gutter__cap.is-jump {
  background: color-mix(in srgb, var(--el-color-primary) 45%, transparent);
}

.wf-gutter__line.is-active,
.wf-gutter__cap.is-active {
  background: var(--el-color-primary);
}

.wf-gutter--dimmed .wf-gutter__line:not(.is-active),
.wf-gutter--dimmed .wf-gutter__cap:not(.is-active) {
  opacity: 0.25;
}
</style>

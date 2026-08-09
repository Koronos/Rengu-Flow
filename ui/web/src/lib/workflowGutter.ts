/**
 * Per-row geometry for the editor's ~28 px connector gutter.
 *
 * {@link assignEdgeLanes} answers *which lane* an edge takes; this answers *what to paint in each
 * row*, which is what a vertical list of independently-sized cards actually needs. There is no
 * canvas and no measurement: every row renders its own slice of every line that crosses it, so a
 * card growing taller cannot desynchronise the rail from the cards.
 *
 * An edge from row `s` to row `t` (always `s < t`, the graph invariant) contributes:
 *
 * ```
 * row s      ├─ span "bottom"  (anchor -> the row's foot)     + cap "start" on an offset lane
 * rows s+1.. │  span "full"    (foot to foot, straight through)
 * row t      ╰─ span "top"     (the row's head -> anchor)     + cap "end" on an offset lane
 * ```
 *
 * Lane 0 — the consecutive link — gets **no cap**: it runs straight past the card's left edge, the
 * `│` of the spec's mockup, and an elbow on it would be visual noise on the one link that needs no
 * explanation. Only a jump (lane >= 1) steps out and needs the horizontal stub back to its card.
 *
 * {@link BADGE_ONLY_LANE} edges produce **nothing**: past three lanes a fourth parallel line is
 * less legible than the `<- from (1)` badge alone, so the badge carries the whole message.
 */

import { BADGE_ONLY_LANE, assignEdgeLanes, edgeKey, type LaneNode } from "./workflowLayout";

/** The vertical extent of one segment within its row, relative to the row's connector anchor. */
export type GutterSpan = "top" | "bottom" | "full";

/** The horizontal stub joining an offset lane to its card; `null` on lane 0. */
export type GutterCap = "start" | "end" | null;

export interface GutterSegment {
  /** {@link edgeKey} of the edge this piece belongs to — the id the hover highlight keys on. */
  key: string;
  lane: number;
  /** Source node id (the `from`). */
  from: string;
  /** Target node id (the node whose `from` this is). */
  to: string;
  span: GutterSpan;
  cap: GutterCap;
}

/**
 * One list of segments per node row, in node order.
 *
 * Rows with nothing crossing them get an empty array rather than being absent, so a component can
 * index straight into the result without a fallback.
 */
export function gutterRows(nodes: readonly LaneNode[]): GutterSegment[][] {
  const lanes = assignEdgeLanes(nodes);
  const rows: GutterSegment[][] = nodes.map(() => []);

  const position = new Map<string, number>();
  nodes.forEach((node, index) => {
    if (!position.has(node.id)) position.set(node.id, index);
  });

  nodes.forEach((node, index) => {
    if (!node.from) return;
    const start = position.get(node.from);
    // Self-links and forward links are not drawable; a valid graph has none, a tolerantly-parsed
    // one can. assignEdgeLanes drops them too, so this only keeps the two in step.
    if (start === undefined || start >= index) return;

    const key = edgeKey(node.from, node.id);
    const lane = lanes[key];
    if (lane === undefined || lane === BADGE_ONLY_LANE) return;

    const base = { key, lane, from: node.from, to: node.id };
    rows[start].push({ ...base, span: "bottom", cap: lane > 0 ? "start" : null });
    for (let row = start + 1; row < index; row += 1) {
      rows[row].push({ ...base, span: "full", cap: null });
    }
    rows[index].push({ ...base, span: "top", cap: lane > 0 ? "end" : null });
  });

  return rows;
}

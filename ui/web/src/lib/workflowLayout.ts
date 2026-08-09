/**
 * Lane assignment for the connector rail in the editor's ~28 px left gutter.
 *
 * The vertical list draws one incoming connector per node. A **consecutive** link (a node reading
 * the card right above it) is a straight segment in lane 0 — the default is silent, no badge is
 * drawn, because the connector already says it. A **jump** (a node reading further back) takes an
 * offset lane that leaves its source, runs down past every skipped card and turns into its
 * target; two jumps that share vertical space must not share a lane or they draw as one line.
 *
 * That is interval packing: each jump owns the closed row range `[sourceRow, targetRow]`, and two
 * edges conflict when their ranges touch at all — including at a shared endpoint, where one line
 * arrives while the other departs and both need the gutter on that row.
 *
 * {@link LANE_COUNT} lanes cover any realistic graph (3-8 nodes, per the spec). An edge that
 * cannot be packed gets {@link BADGE_ONLY_LANE}: no rail is drawn for it and the `⟵ from ①` badge
 * carries the whole message, which is legible where a fourth parallel line would not be.
 */

/** Lane 0 plus two offset lanes for jumps. */
export const LANE_COUNT = 3;

/** "Draw no rail, let the badge speak" — returned when the lanes are full. */
export const BADGE_ONLY_LANE = -1;

/** All this needs of a node: its id and what it reads. Any workflow node satisfies it. */
export interface LaneNode {
  id: string;
  from?: string | null;
}

/** The key an edge is reported under, and the one a component should look its lane up by. */
export function edgeKey(sourceId: string, targetId: string): string {
  return `${sourceId}->${targetId}`;
}

interface Edge {
  key: string;
  start: number;
  end: number;
}

function overlaps(a: Edge, b: Edge): boolean {
  // Closed intervals: sharing a single row is already a collision in a one-line-wide gutter.
  return a.start <= b.end && b.start <= a.end;
}

/**
 * Edge key -> lane index, for every drawable incoming link.
 *
 * Links that cannot be drawn are simply absent: a node with no `from`, a `from` naming a node
 * that is not in the list, and a `from` pointing at itself or forward (which a valid graph never
 * contains, but a tolerantly-parsed one can).
 */
export function assignEdgeLanes(nodes: readonly LaneNode[]): Record<string, number> {
  const position = new Map<string, number>();
  nodes.forEach((node, index) => {
    if (!position.has(node.id)) position.set(node.id, index);
  });

  const lanes: Record<string, number> = {};
  const jumps: Edge[] = [];

  nodes.forEach((node, index) => {
    if (!node.from) return;
    const start = position.get(node.from);
    if (start === undefined || start >= index) return;
    const edge: Edge = { key: edgeKey(node.from, node.id), start, end: index };
    if (index === start + 1) {
      lanes[edge.key] = 0; // consecutive: the straight segment everybody shares
      return;
    }
    jumps.push(edge);
  });

  // Greedy interval packing, earliest source first, into the offset lanes 1..LANE_COUNT-1.
  jumps.sort((a, b) => a.start - b.start || a.end - b.end);
  const packed: Edge[][] = Array.from({ length: LANE_COUNT }, () => []);
  for (const edge of jumps) {
    let lane = BADGE_ONLY_LANE;
    for (let candidate = 1; candidate < LANE_COUNT; candidate += 1) {
      if (packed[candidate].every((other) => !overlaps(edge, other))) {
        lane = candidate;
        break;
      }
    }
    if (lane !== BADGE_ONLY_LANE) packed[lane].push(edge);
    lanes[edge.key] = lane;
  }
  return lanes;
}

/** Whether a link is a jump — i.e. whether the `⟵ from ①` badge should be drawn at all. */
export function isJump(nodes: readonly LaneNode[], nodeId: string): boolean {
  const index = nodes.findIndex((node) => node.id === nodeId);
  if (index < 0) return false;
  const from = nodes[index].from;
  if (!from) return false;
  const start = nodes.findIndex((node) => node.id === from);
  return start >= 0 && start < index - 1;
}

/**
 * The ids of the nodes a jump reads past — the ones that get the faint "③ reads past this step"
 * legend while the badge is hovered.
 */
export function skippedBy(nodes: readonly LaneNode[], nodeId: string): string[] {
  const index = nodes.findIndex((node) => node.id === nodeId);
  if (index < 0) return [];
  const from = nodes[index].from;
  if (!from) return [];
  const start = nodes.findIndex((node) => node.id === from);
  if (start < 0 || start >= index - 1) return [];
  return nodes.slice(start + 1, index).map((node) => node.id);
}

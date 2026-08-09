import { describe, expect, it } from "vitest";
import {
  BADGE_ONLY_LANE,
  LANE_COUNT,
  type LaneNode,
  assignEdgeLanes,
  edgeKey,
  isJump,
  skippedBy,
} from "./workflowLayout";

/** `"n2<-n1"` style shorthand: node id, then the node it reads. */
function list(...spec: [string, string | null][]): LaneNode[] {
  return spec.map(([id, from]) => ({ id, from }));
}

describe("assignEdgeLanes", () => {
  it("puts every consecutive link in lane 0", () => {
    const lanes = assignEdgeLanes(list(["n1", null], ["n2", "n1"], ["n3", "n2"], ["n4", "n3"]));
    expect(lanes).toEqual({
      [edgeKey("n1", "n2")]: 0,
      [edgeKey("n2", "n3")]: 0,
      [edgeKey("n3", "n4")]: 0,
    });
  });

  it("gives a jump an offset lane and leaves lane 0 to the chain", () => {
    // n1 → n2 → n3, and n3 also skipped by n4 reading n1.
    const lanes = assignEdgeLanes(list(["n1", null], ["n2", "n1"], ["n3", "n2"], ["n4", "n1"]));
    expect(lanes[edgeKey("n2", "n3")]).toBe(0);
    expect(lanes[edgeKey("n1", "n4")]).toBe(1);
  });

  it("separates two overlapping jumps", () => {
    // n1→n3 spans rows 0-2, n2→n4 spans rows 1-3: they share rows 1 and 2.
    const lanes = assignEdgeLanes(list(["n1", null], ["n2", "n1"], ["n3", "n1"], ["n4", "n2"]));
    expect(lanes[edgeKey("n1", "n3")]).toBe(1);
    expect(lanes[edgeKey("n2", "n4")]).toBe(2);
  });

  it("separates nested jumps, outermost first", () => {
    // n1→n5 (rows 0-4) contains n2→n4 (rows 1-3).
    const lanes = assignEdgeLanes(
      list(["n1", null], ["n2", "n1"], ["n3", "n2"], ["n4", "n2"], ["n5", "n1"]),
    );
    expect(lanes[edgeKey("n1", "n5")]).toBe(1);
    expect(lanes[edgeKey("n2", "n4")]).toBe(2);
    expect(lanes[edgeKey("n2", "n3")]).toBe(0);
  });

  it("reuses a lane for jumps that do not share any row", () => {
    // n1→n3 spans rows 0-2; n4→n6 spans rows 3-5. Nothing in common, same lane.
    const lanes = assignEdgeLanes(
      list(["n1", null], ["n2", "n1"], ["n3", "n1"], ["n4", "n3"], ["n5", "n4"], ["n6", "n4"]),
    );
    expect(lanes[edgeKey("n1", "n3")]).toBe(1);
    expect(lanes[edgeKey("n4", "n6")]).toBe(1);
  });

  it("treats jumps that meet at one node as a collision", () => {
    // n1→n3 ends on row 2 and n3→n5 starts there: one line arrives while the other departs.
    const lanes = assignEdgeLanes(
      list(["n1", null], ["n2", "n1"], ["n3", "n1"], ["n4", "n3"], ["n5", "n3"]),
    );
    expect(lanes[edgeKey("n1", "n3")]).toBe(1);
    expect(lanes[edgeKey("n3", "n5")]).toBe(2);
  });

  it("collapses to badge-only once the lanes are full", () => {
    // Three jumps all crossing row 3; only two offset lanes exist.
    const lanes = assignEdgeLanes(
      list(
        ["n1", null],
        ["n2", "n1"],
        ["n3", "n1"],
        ["n4", "n1"],
        ["n5", "n2"],
        ["n6", "n3"],
        ["n7", "n4"],
      ),
    );
    expect(lanes[edgeKey("n1", "n3")]).toBe(1);
    expect(lanes[edgeKey("n1", "n4")]).toBe(2);
    expect(lanes[edgeKey("n2", "n5")]).toBe(BADGE_ONLY_LANE);
    // The overflow is local: a later, disjoint jump still gets a rail.
    expect(lanes[edgeKey("n4", "n7")]).toBe(1);
    expect(LANE_COUNT).toBe(3);
  });

  it("never assigns two overlapping edges to the same offset lane", () => {
    const nodes = list(
      ["n1", null],
      ["n2", "n1"],
      ["n3", "n1"],
      ["n4", "n2"],
      ["n5", "n3"],
      ["n6", "n5"],
      ["n7", "n1"],
    );
    const position = new Map(nodes.map((node, index) => [node.id, index]));
    const lanes = assignEdgeLanes(nodes);
    const spans = Object.entries(lanes)
      .filter(([, lane]) => lane > 0)
      .map(([key, lane]) => {
        const [from, to] = key.split("->");
        return { lane, start: position.get(from)!, end: position.get(to)! };
      });
    for (const a of spans) {
      for (const b of spans) {
        if (a === b || a.lane !== b.lane) continue;
        expect(a.start <= b.end && b.start <= a.end).toBe(false);
      }
    }
  });

  it("omits links that cannot be drawn", () => {
    expect(assignEdgeLanes(list(["n1", null]))).toEqual({});
    expect(assignEdgeLanes(list(["n1", "ghost"], ["n2", "n1"]))).toEqual({
      [edgeKey("n1", "n2")]: 0,
    });
    // Self-reference and a forward reference: invalid graphs the tolerant parser still hands us.
    expect(assignEdgeLanes(list(["n1", "n1"], ["n2", "n3"], ["n3", null]))).toEqual({});
  });

  it("handles an empty list", () => {
    expect(assignEdgeLanes([])).toEqual({});
  });
});

describe("isJump / skippedBy", () => {
  const nodes = list(["n1", null], ["n2", "n1"], ["n3", "n2"], ["n4", "n1"]);

  it("badges only the non-consecutive links", () => {
    expect(isJump(nodes, "n2")).toBe(false);
    expect(isJump(nodes, "n3")).toBe(false);
    expect(isJump(nodes, "n4")).toBe(true);
    expect(isJump(nodes, "n1")).toBe(false);
    expect(isJump(nodes, "nope")).toBe(false);
  });

  it("names the cards a jump reads past", () => {
    expect(skippedBy(nodes, "n4")).toEqual(["n2", "n3"]);
    expect(skippedBy(nodes, "n3")).toEqual([]);
    expect(skippedBy(nodes, "nope")).toEqual([]);
  });
});
